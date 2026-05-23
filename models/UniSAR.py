import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from utils import const

from .BaseModel import BaseModel
from .layers import FullyConnectedLayer, feature_align, PositionalEmbedding, PLE_layer


class LatentIntentDiscovery(nn.Module):
    def __init__(self, emb_size, intent_num, intent_heads, dropout) -> None:
        super().__init__()
        self.intent_num = intent_num
        self.intent_slots = nn.Parameter(torch.randn(intent_num, emb_size))
        nn.init.xavier_normal_(self.intent_slots)
        self.slot_attention = nn.MultiheadAttention(emb_size,
                                                    intent_heads,
                                                    dropout=dropout,
                                                    batch_first=True)

    def forward(self, seq_emb: torch.Tensor, seq_mask: torch.Tensor):
        batch_size = seq_emb.size(0)
        slots = self.intent_slots.unsqueeze(0).expand(batch_size, -1, -1)

        safe_mask = seq_mask.clone()
        all_masked = safe_mask.all(dim=1)
        if all_masked.any():
            safe_mask[all_masked] = False

        intents, _ = self.slot_attention(slots,
                                         seq_emb,
                                         seq_emb,
                                         key_padding_mask=safe_mask,
                                         need_weights=False)
        if all_masked.any():
            intents = torch.where(all_masked[:, None, None],
                                  slots,
                                  intents)
        return intents


class UniSAR(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=2)

        parser.add_argument('--q_i_cl_temp', type=float, default=0.5)
        parser.add_argument('--q_i_cl_weight', type=float, default=0.001)

        parser.add_argument('--his_cl_temp', type=float, default=0.1)
        parser.add_argument('--his_cl_weight', type=float, default=0.1)

        parser.add_argument('--pred_hid_units',
                            type=List,
                            default=[200, 80, 1])

        parser.add_argument('--intent_num', type=int, default=8)
        parser.add_argument('--intent_heads', type=int, default=2)
        parser.add_argument('--intent_dropout', type=float, default=0.1)
        parser.add_argument('--intent_temp', type=float, default=0.5)
        parser.add_argument('--intent_var_min', type=float, default=1e-4)
        parser.add_argument('--intent_diversity_weight',
                            type=float,
                            default=0.01)
        parser.add_argument('--intent_diversity_margin',
                            type=float,
                            default=0.2)

        parser.add_argument('--belief_init_var', type=float, default=1.0)
        parser.add_argument('--belief_init_mass', type=float, default=1.0)
        parser.add_argument('--belief_prior_weight', type=float, default=1.0)
        parser.add_argument('--belief_drift_decay', type=float, default=0.98)

        parser.add_argument('--intent_bias_scale', type=float, default=1.0)
        parser.add_argument('--cf_temp', type=float, default=1.0)
        parser.add_argument('--cf_bias_scale', type=float, default=1.0)

        return BaseModel.parse_model_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size

        self.intent_num = args.intent_num
        self.intent_temp = args.intent_temp
        self.intent_var_min = args.intent_var_min
        self.intent_diversity_weight = args.intent_diversity_weight
        self.intent_diversity_margin = args.intent_diversity_margin
        self.belief_init_var = args.belief_init_var
        self.belief_init_mass = args.belief_init_mass
        self.belief_prior_weight = args.belief_prior_weight
        self.belief_drift_decay = args.belief_drift_decay
        self.cf_temp = args.cf_temp

        self.latent_intent_discovery = LatentIntentDiscovery(
            self.item_size, self.intent_num, args.intent_heads,
            args.intent_dropout)

        self.src_pos = PositionalEmbedding(const.max_src_session_his_len,
                                           self.item_size)
        self.rec_pos = PositionalEmbedding(const.max_rec_his_len,
                                           self.item_size)
        self.global_pos_emb = PositionalEmbedding(
            const.max_rec_his_len + const.max_src_session_his_len,
            self.item_size)

        self.rec_transformer = Transformer(emb_size=self.item_size,
                                           num_heads=self.num_heads,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout,
                                           intent_bias_scale=args.intent_bias_scale,
                                           cf_bias_scale=args.cf_bias_scale)
        self.src_transformer = Transformer(emb_size=self.item_size,
                                           num_heads=self.num_heads,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout,
                                           intent_bias_scale=args.intent_bias_scale,
                                           cf_bias_scale=args.cf_bias_scale)
        self.global_transformer = Transformer(emb_size=self.item_size,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_layers,
                                              dropout=self.dropout,
                                              intent_bias_scale=args.intent_bias_scale,
                                              cf_bias_scale=args.cf_bias_scale)

        self.q_i_cl_temp = args.q_i_cl_temp
        self.q_i_cl_weight = args.q_i_cl_weight
        if self.q_i_cl_weight > 0:
            self.query_item_alignment = True
            self.feature_alignment = feature_align(self.q_i_cl_temp,
                                                   self.item_size)

        self.his_cl_temp = args.his_cl_temp
        self.his_cl_weight = args.his_cl_weight
        if self.his_cl_weight > 0:
            self.rec_his_cl = TransAlign(batch_size=self.batch_size,
                                         hidden_dim=self.item_size,
                                         device=self.device,
                                         infoNCE_temp=self.his_cl_temp)
            self.src_his_cl = TransAlign(batch_size=self.batch_size,
                                         hidden_dim=self.item_size,
                                         device=self.device,
                                         infoNCE_temp=self.his_cl_temp)

        self.transformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.item_size,
            nhead=self.num_heads,
            dim_feedforward=self.item_size,
            dropout=self.dropout,
            batch_first=True)

        self.src_cross_fusion = nn.TransformerDecoder(
            self.transformerDecoderLayer, num_layers=self.num_layers)
        self.rec_cross_fusion = nn.TransformerDecoder(
            self.transformerDecoderLayer, num_layers=self.num_layers)

        self.rec_his_attn_pooling = Target_Attention(self.item_size,
                                                     self.item_size)
        self.src_his_attn_pooling = Target_Attention(self.item_size,
                                                     self.item_size)

        self.rec_query = torch.nn.parameter.Parameter(torch.randn(
            (1, self.query_size), requires_grad=True),
                                                      requires_grad=True)
        nn.init.xavier_normal_(self.rec_query)

        self.hidden_unit = args.pred_hid_units

        input_dim = 3 * self.item_size + self.user_size + self.query_size
        self.ple_layer = PLE_layer(orig_input_dim=input_dim,
                                   bottom_mlp_dims=[64],
                                   tower_mlp_dims=[128, 64],
                                   task_num=2,
                                   shared_expert_num=4,
                                   specific_expert_num=4,
                                   dropout=self.dropout)
        self.rec_fc_layer = FullyConnectedLayer(input_size=64,
                                                hidden_unit=self.hidden_unit,
                                                batch_norm=False,
                                                sigmoid=True,
                                                activation='relu',
                                                dropout=self.dropout)
        self.src_fc_layer = FullyConnectedLayer(input_size=64,
                                                hidden_unit=self.hidden_unit,
                                                batch_norm=False,
                                                sigmoid=True,
                                                activation='relu',
                                                dropout=self.dropout)

        self.loss_fn = nn.BCELoss()
        self._init_weights()
        self.to(self.device)

    def src_feat_process(self, src_feat):
        query_emb, q_click_item_emb, click_item_mask = src_feat

        q_i_align_used = [query_emb, click_item_mask, q_click_item_emb]

        mean_click_item_emb = torch.sum(torch.mul(
            q_click_item_emb, click_item_mask.unsqueeze(-1)),
                                        dim=-2)  # batch, max_src_len, dim
        mean_click_item_emb = mean_click_item_emb / (torch.max(
            click_item_mask.sum(-1, keepdim=True),
            torch.ones_like(click_item_mask.sum(-1, keepdim=True))))
        query_his_emb = query_emb
        click_item_his_emb = mean_click_item_emb

        return query_his_emb + click_item_his_emb, q_i_align_used

    def get_all_his_emb(self, all_his, all_his_type):
        rec_his = torch.masked_fill(all_his, all_his_type != 1, 0)
        rec_his_emb = self.session_embedding.get_item_emb(rec_his)
        rec_his_emb = torch.masked_fill(rec_his_emb,
                                        (all_his_type != 1).unsqueeze(-1), 0)

        src_session_his = torch.masked_fill(all_his, all_his_type != 2, 0)
        src_his_emb, q_i_align_used = self.src_feat_process(
            self.session_embedding(src_session_his))
        src_his_emb = torch.masked_fill(src_his_emb,
                                        (all_his_type != 2).unsqueeze(-1), 0)

        all_his_emb = rec_his_emb + src_his_emb
        all_his_mask = torch.where(all_his == 0, 1, 0).bool()

        return all_his_emb, all_his_mask, q_i_align_used

    def repeat_feat(self, feature_list, items_emb):
        repeat_feature_list = [
            torch.repeat_interleave(feat, items_emb.size(1), dim=0)
            for feat in feature_list
        ]
        items_emb = items_emb.reshape(-1, items_emb.size(-1))

        return repeat_feature_list, items_emb

    def mean_pooling(self, output, his_len):
        return torch.sum(output, dim=1) / his_len.unsqueeze(-1)

    def split_rec_src(self, all_his_emb, all_his_type):
        rec_his_emb = torch.masked_select(
            all_his_emb, (all_his_type == 1).unsqueeze(-1)).reshape(
                (all_his_emb.shape[0], const.max_rec_his_len,
                 all_his_emb.shape[2]))
        src_his_emb = torch.masked_select(
            all_his_emb, (all_his_type == 2).unsqueeze(-1)).reshape(
                (all_his_emb.shape[0], const.max_src_session_his_len,
                 all_his_emb.shape[2]))
        return rec_his_emb, src_his_emb

    def split_rec_src_trace(self, trace, all_his_type):
        rec_trace = torch.masked_select(
            trace, (all_his_type == 1).unsqueeze(-1)).reshape(
                (trace.shape[0], const.max_rec_his_len, trace.shape[2]))
        src_trace = torch.masked_select(
            trace, (all_his_type == 2).unsqueeze(-1)).reshape(
                (trace.shape[0], const.max_src_session_his_len,
                 trace.shape[2]))
        return rec_trace, src_trace

    def split_rec_src_value(self, value, all_his_type):
        rec_value = torch.masked_select(value, all_his_type == 1).reshape(
            (value.shape[0], const.max_rec_his_len))
        src_value = torch.masked_select(value, all_his_type == 2).reshape(
            (value.shape[0], const.max_src_session_his_len))
        return rec_value, src_value

    def compute_intent_state(self, seq_emb, seq_mask):
        intents = self.latent_intent_discovery(seq_emb, seq_mask)
        assign_logits = torch.matmul(seq_emb, intents.transpose(-2, -1))
        assign_logits = assign_logits / self.intent_temp
        prior_assign = F.softmax(assign_logits, dim=-1)
        prior_assign = prior_assign.masked_fill(seq_mask.unsqueeze(-1), 0)

        norm_intents = F.normalize(intents, dim=-1)
        intent_sim = torch.matmul(norm_intents, norm_intents.transpose(-2, -1))
        eye = torch.eye(self.intent_num,
                        dtype=torch.bool,
                        device=intent_sim.device).unsqueeze(0).expand(
                            intent_sim.size(0), -1, -1)
        off_diag_sim = intent_sim.masked_select(~eye)
        intent_reg = F.relu(
            off_diag_sim - self.intent_diversity_margin).mean()

        diagnostics = {}
        return intents, prior_assign, intent_reg, diagnostics

    def compute_belief_trace(self,
                             seq_emb,
                             intents,
                             prior_assign,
                             seq_mask,
                             update_mask=None):
        batch_size, seq_len, _ = seq_emb.shape
        valid = ~seq_mask
        if update_mask is None:
            update_mask = valid

        mu = intents
        var = torch.full_like(mu, self.belief_init_var)
        mass = torch.full((batch_size, self.intent_num),
                          self.belief_init_mass,
                          dtype=seq_emb.dtype,
                          device=seq_emb.device)

        posterior_trace = seq_emb.new_zeros(batch_size, seq_len,
                                            self.intent_num)
        confidence_trace = seq_emb.new_zeros(batch_size, seq_len)

        for t in range(seq_len):
            valid_t = valid[:, t]
            can_update_t = valid_t & update_mask[:, t]
            x_t = seq_emb[:, t, :]

            delta = x_t.unsqueeze(1) - mu
            cost = (delta.pow(2) / var.clamp_min(
                self.intent_var_min)).mean(dim=-1)
            log_likelihood = -0.5 * cost
            log_prior = prior_assign[:, t, :].clamp_min(1e-8).log()
            score = log_likelihood + self.belief_prior_weight * log_prior
            posterior_t = F.softmax(score, dim=-1)
            posterior_t = posterior_t.masked_fill(~valid_t.unsqueeze(-1), 0)
            posterior_trace[:, t, :] = posterior_t

            sigma = var.clamp_min(self.intent_var_min).sqrt().mean(dim=-1)
            expected_sigma = (posterior_t * sigma).sum(dim=-1)
            confidence_t = 1.0 / (1.0 + expected_sigma)
            confidence_t = confidence_t.masked_fill(~valid_t, 0)
            confidence_trace[:, t] = confidence_t

            effective_old_mass = self.belief_drift_decay * mass
            update_weight = posterior_t.masked_fill(
                ~can_update_t.unsqueeze(-1), 0)
            new_mass = effective_old_mass + update_weight
            new_mass_clamped = new_mass.clamp_min(1e-8)

            old_second = var + mu.pow(2)
            new_mu = (effective_old_mass.unsqueeze(-1) * mu +
                      update_weight.unsqueeze(-1) * x_t.unsqueeze(1)
                      ) / new_mass_clamped.unsqueeze(-1)
            new_second = (
                effective_old_mass.unsqueeze(-1) * old_second +
                update_weight.unsqueeze(-1) * x_t.pow(2).unsqueeze(1)
            ) / new_mass_clamped.unsqueeze(-1)
            new_var = (new_second - new_mu.pow(2)).clamp_min(
                self.intent_var_min)

            update_rows = can_update_t[:, None, None]
            mu = torch.where(update_rows, new_mu, mu)
            var = torch.where(update_rows, new_var, var)
            mass = torch.where(can_update_t[:, None], new_mass, mass)

        diagnostics = {}
        return {
            'posterior_trace': posterior_trace,
            'confidence_trace': confidence_trace,
            'final_mu': mu,
            'final_var': var,
            'final_mass': mass,
            'diagnostics': diagnostics
        }

    def compute_belief_traces(self, seq_emb, intents, prior_assign, seq_mask,
                              update_masks):
        num_states = len(update_masks)
        batch_size, seq_len, _ = seq_emb.shape
        valid = ~seq_mask
        stacked_update_masks = torch.stack(update_masks, dim=0)

        mu = intents.unsqueeze(0).expand(num_states, -1, -1, -1)
        var = torch.full_like(mu, self.belief_init_var)
        mass = torch.full((num_states, batch_size, self.intent_num),
                          self.belief_init_mass,
                          dtype=seq_emb.dtype,
                          device=seq_emb.device)

        posterior_trace = seq_emb.new_zeros(num_states, batch_size, seq_len,
                                            self.intent_num)
        confidence_trace = seq_emb.new_zeros(num_states, batch_size, seq_len)

        for t in range(seq_len):
            valid_t = valid[:, t]
            can_update_t = valid_t.unsqueeze(0) & stacked_update_masks[:, :, t]
            x_t = seq_emb[:, t, :]

            delta = x_t.unsqueeze(0).unsqueeze(2) - mu
            cost = (delta.pow(2) / var.clamp_min(
                self.intent_var_min)).mean(dim=-1)
            log_likelihood = -0.5 * cost
            log_prior = prior_assign[:, t, :].clamp_min(1e-8).log()
            score = log_likelihood + self.belief_prior_weight * log_prior
            posterior_t = F.softmax(score, dim=-1)
            posterior_t = posterior_t.masked_fill(
                ~valid_t.unsqueeze(0).unsqueeze(-1), 0)
            posterior_trace[:, :, t, :] = posterior_t

            sigma = var.clamp_min(self.intent_var_min).sqrt().mean(dim=-1)
            expected_sigma = (posterior_t * sigma).sum(dim=-1)
            confidence_t = 1.0 / (1.0 + expected_sigma)
            confidence_t = confidence_t.masked_fill(~valid_t.unsqueeze(0), 0)
            confidence_trace[:, :, t] = confidence_t

            effective_old_mass = self.belief_drift_decay * mass
            update_weight = posterior_t.masked_fill(
                ~can_update_t.unsqueeze(-1), 0)
            new_mass = effective_old_mass + update_weight
            new_mass_clamped = new_mass.clamp_min(1e-8)

            old_second = var + mu.pow(2)
            new_mu = (effective_old_mass.unsqueeze(-1) * mu +
                      update_weight.unsqueeze(-1) *
                      x_t.unsqueeze(0).unsqueeze(2)
                      ) / new_mass_clamped.unsqueeze(-1)
            new_second = (
                effective_old_mass.unsqueeze(-1) * old_second +
                update_weight.unsqueeze(-1) *
                x_t.pow(2).unsqueeze(0).unsqueeze(2)
            ) / new_mass_clamped.unsqueeze(-1)
            new_var = (new_second - new_mu.pow(2)).clamp_min(
                self.intent_var_min)

            update_rows = can_update_t.unsqueeze(-1).unsqueeze(-1)
            mu = torch.where(update_rows, new_mu, mu)
            var = torch.where(update_rows, new_var, var)
            mass = torch.where(can_update_t.unsqueeze(-1), new_mass, mass)

        return [{
            'posterior_trace': posterior_trace[i],
            'confidence_trace': confidence_trace[i],
            'final_mu': mu[i],
            'final_var': var[i],
            'final_mass': mass[i],
            'diagnostics': {}
        } for i in range(num_states)]

    def compute_source_counterfactual(self, p_full, p_no_rec, p_no_src,
                                      all_his_mask):
        p = p_full.clamp_min(1e-8)
        rec_effect = (p * (p.log() - p_no_rec.clamp_min(1e-8).log())).sum(
            dim=-1)
        src_effect = (p * (p.log() - p_no_src.clamp_min(1e-8).log())).sum(
            dim=-1)
        rec_effect = rec_effect.masked_fill(all_his_mask, 0)
        src_effect = src_effect.masked_fill(all_his_mask, 0)

        cf_logits = torch.stack([rec_effect, src_effect], dim=-1)
        cf_source_gate = F.softmax(cf_logits / self.cf_temp, dim=-1)
        cf_source_gate = cf_source_gate.masked_fill(all_his_mask.unsqueeze(-1),
                                                   0)
        diagnostics = {}
        return cf_source_gate, rec_effect, src_effect, diagnostics

    def build_rel_sar_state(self, all_his_emb, all_his_mask, all_his_type):
        all_intents, all_prior_assign, intent_reg, _ = \
            self.compute_intent_state(all_his_emb, all_his_mask)

        valid = ~all_his_mask
        rec_update = valid & (all_his_type == 1)
        src_update = valid & (all_his_type == 2)

        full_belief, no_rec_belief, no_src_belief = \
            self.compute_belief_traces(
                all_his_emb, all_intents, all_prior_assign, all_his_mask,
                [valid, src_update, rec_update])

        full_posterior = full_belief['posterior_trace']
        full_confidence = full_belief['confidence_trace']
        cf_source_gate, rec_effect, src_effect, _ = \
            self.compute_source_counterfactual(
                full_posterior, no_rec_belief['posterior_trace'],
                no_src_belief['posterior_trace'], all_his_mask)

        valid_float = valid.float()
        denom = valid_float.sum().clamp_min(1.0)
        entropy = -(full_posterior.clamp_min(1e-8) *
                    full_posterior.clamp_min(1e-8).log()).sum(dim=-1)
        sigma = full_belief['final_var'].clamp_min(
            self.intent_var_min).sqrt().mean()
        diagnostics = {
            'belief_entropy_mean': (entropy * valid_float).sum() / denom,
            'belief_sigma_mean': sigma,
            'belief_confidence_mean':
            (full_confidence * valid_float).sum() / denom,
            'cf_rec_effect_mean': (rec_effect * valid_float).sum() / denom,
            'cf_src_effect_mean': (src_effect * valid_float).sum() / denom,
            'cf_rec_gate_mean':
            (cf_source_gate[:, :, 0] * valid_float).sum() / denom,
            'cf_src_gate_mean':
            (cf_source_gate[:, :, 1] * valid_float).sum() / denom
        }

        return {
            'prior_assign': all_prior_assign,
            'posterior': full_posterior,
            'confidence': full_confidence,
            'source_gate': cf_source_gate,
            'intent_reg': intent_reg,
            'diagnostics': diagnostics
        }

    def forward(self, user, all_his, all_his_type, items_emb, domain):
        user_emb = self.session_embedding.get_user_emb(user)

        all_his_emb, all_his_mask, q_i_align_used = self.get_all_his_emb(
            all_his, all_his_type)

        rec_his_mask = torch.masked_select(all_his_mask,
                                           (all_his_type == 1)).reshape(
                                               (all_his_emb.shape[0],
                                                const.max_rec_his_len))
        src_his_mask = torch.masked_select(all_his_mask,
                                           (all_his_type == 2)).reshape(
                                               (all_his_emb.shape[0],
                                               const.max_src_session_his_len))

        rel_sar_state = self.build_rel_sar_state(all_his_emb, all_his_mask,
                                                 all_his_type)
        rec_posterior, src_posterior = self.split_rec_src_trace(
            rel_sar_state['posterior'], all_his_type)
        rec_confidence, src_confidence = self.split_rec_src_value(
            rel_sar_state['confidence'], all_his_type)

        all_his_emb_w_pos = all_his_emb + self.global_pos_emb(all_his_emb)

        global_mask = all_his_type[:, :, None] == all_his_type[:, None, :]

        global_encoded = self.global_transformer(
            all_his_emb_w_pos,
            all_his_mask,
            global_mask,
            intent_assign=rel_sar_state['posterior'],
            belief_confidence=rel_sar_state['confidence'],
            source_gate=rel_sar_state['source_gate'],
            token_type=all_his_type)
        src2rec, rec2src = self.split_rec_src(global_encoded, all_his_type)

        rec_his_emb, src_his_emb = self.split_rec_src(all_his_emb,
                                                      all_his_type)
        rec_his_emb_w_pos = rec_his_emb + self.rec_pos(rec_his_emb)
        src_his_emb_w_pos = src_his_emb + self.src_pos(src_his_emb)

        rec2rec = self.rec_transformer(rec_his_emb_w_pos,
                                       rec_his_mask,
                                       intent_assign=rec_posterior,
                                       belief_confidence=rec_confidence)
        src2src = self.src_transformer(src_his_emb_w_pos,
                                       src_his_mask,
                                       intent_assign=src_posterior,
                                       belief_confidence=src_confidence)

        rec_fusion_decoded = self.rec_cross_fusion(
            tgt=rec2rec,
            memory=src2rec,
            tgt_key_padding_mask=rec_his_mask,
            memory_key_padding_mask=rec_his_mask)

        src_fusion_decoded = self.src_cross_fusion(
            tgt=src2src,
            memory=rec2src,
            tgt_key_padding_mask=src_his_mask,
            memory_key_padding_mask=src_his_mask)

        his_cl_used = [
            src2rec, rec2rec, rec_his_mask, rec2src, src2src, src_his_mask
        ]

        if domain == 'rec':
            feature_list = [
                rec_fusion_decoded, rec_his_mask, src_fusion_decoded,
                src_his_mask, user_emb
            ]
            repeat_feature_list, items_emb = self.repeat_feat(
                feature_list, items_emb)
            rec_fusion_decoded, rec_his_mask,\
                src_fusion_decoded, src_his_mask,\
                user_emb = repeat_feature_list

        rec_fusion = self.rec_his_attn_pooling(rec_fusion_decoded, items_emb,
                                               rec_his_mask)
        src_fusion = self.src_his_attn_pooling(src_fusion_decoded, items_emb,
                                               src_his_mask)

        user_feats = [rec_fusion, src_fusion, user_emb]

        return user_feats, q_i_align_used, his_cl_used, rel_sar_state

    def inter_pred(self, user_feats, item_emb, domain, query_emb=None):
        assert domain in ["rec", "src"]

        rec_interest, src_interest, user_emb = user_feats

        if domain == "rec":
            item_emb = item_emb.reshape(-1, item_emb.size(-1))

            output = self.ple_layer(
                torch.cat([
                    rec_interest, src_interest, item_emb, user_emb,
                    self.rec_query.expand(item_emb.shape[0], -1)
                ], -1))[0]

            return self.rec_fc_layer(output)

        elif domain == "src":
            if item_emb.dim() == 3:
                [query_emb], item_emb = self.repeat_feat([query_emb], item_emb)

            output = self.ple_layer(
                torch.cat([
                    rec_interest, src_interest, item_emb, user_emb, query_emb
                ], -1))[1]
            return self.src_fc_layer(output)

    def rec_loss(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used, rel_sar_state = self.forward(
            user, all_his, all_his_type, items_emb, domain='rec')

        logits = self.inter_pred(user_feats, items_emb, domain="rec").reshape(
            (batch_size, -1))
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))

        total_loss = self.loss_fn(logits, labels)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()

        if self.q_i_cl_weight > 0:
            align_neg_item, align_neg_query = inputs['align_neg_item'], inputs[
                'align_neg_query']
            query_emb, click_item_mask, q_click_item_emb = q_i_align_used

            align_neg_items_emb = self.session_embedding.get_item_emb(
                align_neg_item)
            align_neg_querys_emb = self.session_embedding.get_query_emb(
                align_neg_query)
            align_loss = self.feature_alignment(
                [align_neg_items_emb, align_neg_querys_emb], query_emb,
                click_item_mask, q_click_item_emb)
            loss_dict['q_i_cl_loss'] = align_loss.clone()

            total_loss += self.q_i_cl_weight * align_loss

        if self.his_cl_weight > 0:
            src2rec, rec2rec, rec_his_mask,\
                rec2src, src2src, src_his_mask = his_cl_used
            rec_his_cl_loss = self.rec_his_cl(src2rec, rec2rec, rec_his_mask)

            src_his_cl_loss = self.src_his_cl(rec2src, src2src, src_his_mask)

            his_cl_loss = rec_his_cl_loss + src_his_cl_loss
            loss_dict['his_cl_loss'] = his_cl_loss.clone()

            total_loss += self.his_cl_weight * his_cl_loss

        loss_dict['intent_reg'] = rel_sar_state['intent_reg'].clone()
        total_loss += self.intent_diversity_weight * rel_sar_state[
            'intent_reg']
        for key, value in rel_sar_state['diagnostics'].items():
            loss_dict[key] = value.detach()

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def rec_predict(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used, _ = self.forward(
            user, all_his, all_his_type, items_emb, domain='rec')

        logits = self.inter_pred(user_feats, items_emb, domain="rec").reshape(
            (batch_size, -1))
        return logits

    def src_loss(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        query = inputs['query']
        query_emb = self.session_embedding.get_query_emb(query)

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used, rel_sar_state = self.forward(
            user, all_his, all_his_type, items_emb, domain='rec')

        logits = self.inter_pred(user_feats,
                                 items_emb,
                                 domain="src",
                                 query_emb=query_emb).reshape((batch_size, -1))
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))

        total_loss = self.loss_fn(logits, labels)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()

        if self.q_i_cl_weight > 0:
            align_neg_item, align_neg_query = inputs['align_neg_item'], inputs[
                'align_neg_query']
            query_emb, click_item_mask, q_click_item_emb = q_i_align_used

            align_neg_items_emb = self.session_embedding.get_item_emb(
                align_neg_item)
            align_neg_querys_emb = self.session_embedding.get_query_emb(
                align_neg_query)
            align_loss = self.feature_alignment(
                [align_neg_items_emb, align_neg_querys_emb], query_emb,
                click_item_mask, q_click_item_emb)
            loss_dict['q_i_cl_loss'] = align_loss.clone()

            total_loss += self.q_i_cl_weight * align_loss

        if self.his_cl_weight > 0:
            src2rec, rec2rec, rec_his_mask,\
                rec2src, src2src, src_his_mask = his_cl_used

            rec_his_cl_loss = self.rec_his_cl(src2rec, rec2rec, rec_his_mask)

            src_his_cl_loss = self.src_his_cl(rec2src, src2src, src_his_mask)

            his_cl_loss = rec_his_cl_loss + src_his_cl_loss
            loss_dict['his_cl_loss'] = his_cl_loss.clone()

            total_loss += self.his_cl_weight * his_cl_loss

        loss_dict['intent_reg'] = rel_sar_state['intent_reg'].clone()
        total_loss += self.intent_diversity_weight * rel_sar_state[
            'intent_reg']
        for key, value in rel_sar_state['diagnostics'].items():
            loss_dict[key] = value.detach()

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def src_predict(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        query = inputs['query']
        query_emb = self.session_embedding.get_query_emb(query)

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used, _ = self.forward(
            user, all_his, all_his_type, items_emb, domain='rec')

        logits = self.inter_pred(user_feats,
                                 items_emb,
                                 domain="src",
                                 query_emb=query_emb).reshape((batch_size, -1))
        return logits


class Target_Attention(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()

        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_emb, target, mask):
        score = torch.matmul(seq_emb, self.W)
        score = torch.matmul(score, target.unsqueeze(-1))

        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(all_score.transpose(-2, -1))
        all_vec = torch.matmul(all_weight, seq_emb).squeeze(1)

        return all_vec


class TransAlign(nn.Module):
    def __init__(self, batch_size, hidden_dim, device, infoNCE_temp) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.infoNCE_temp = nn.Parameter(torch.ones([]) * infoNCE_temp)
        self.weight_matrix = nn.Parameter(torch.randn(
            (hidden_dim, hidden_dim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.cl_loss_func = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(self.batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, same_his: torch.Tensor, diff_his: torch.Tensor,
                his_mask: torch.Tensor):
        same_his_emb = same_his.masked_fill(his_mask.unsqueeze(2), 0)
        same_his_sum = same_his_emb.sum(dim=1)
        same_his_mean = same_his_sum / \
            (~his_mask).sum(dim=1, keepdim=True)

        diff_his_emb = diff_his.masked_fill(his_mask.unsqueeze(2), 0)
        diff_his_sum = diff_his_emb.sum(dim=1)
        diff_his_mean = diff_his_sum / \
            (~his_mask).sum(dim=1, keepdim=True)

        batch_size = same_his_mean.size(0)
        N = 2 * batch_size

        z = torch.cat([same_his_mean.squeeze(),
                       diff_his_mean.squeeze()],
                      dim=0)
        sim = torch.mm(torch.mm(z, self.weight_matrix), z.T)
        sim = torch.tanh(sim) / self.infoNCE_temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)

        return info_nce_loss


class IntentSourceSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, intent_bias_scale,
                 cf_bias_scale) -> None:
        super().__init__()
        assert emb_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.intent_bias_scale = intent_bias_scale
        self.cf_bias_scale = cf_bias_scale

        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads,
                         self.head_dim).transpose(1, 2)

    def _build_attention_bias(self, seq_len, intent_assign,
                              belief_confidence, source_gate, token_type):
        bias = None
        if intent_assign is not None and belief_confidence is not None:
            intent_sim = torch.matmul(intent_assign,
                                      intent_assign.transpose(-2, -1))
            intent_center = 1.0 / intent_assign.size(-1)
            conf_i = belief_confidence.unsqueeze(-1)
            conf_j = belief_confidence.unsqueeze(1)
            pair_confidence = (conf_i * conf_j).clamp_min(0).sqrt()
            intent_bias = (intent_sim - intent_center) * pair_confidence
            bias = self.intent_bias_scale * intent_bias

        if (source_gate is not None and token_type is not None
                and belief_confidence is not None):
            rec_support = source_gate[:, :, 0].unsqueeze(-1).expand(
                -1, -1, seq_len)
            src_support = source_gate[:, :, 1].unsqueeze(-1).expand(
                -1, -1, seq_len)
            key_is_rec = (token_type == 1).unsqueeze(1).expand(
                -1, seq_len, -1)
            key_is_src = (token_type == 2).unsqueeze(1).expand(
                -1, seq_len, -1)
            source_support = torch.zeros_like(rec_support)
            source_support = torch.where(key_is_rec, rec_support,
                                         source_support)
            source_support = torch.where(key_is_src, src_support,
                                         source_support)
            source_bias = (source_support - 0.5) * belief_confidence.unsqueeze(
                -1)
            source_bias = source_bias.masked_fill(~(key_is_rec | key_is_src),
                                                  0)
            if bias is None:
                bias = self.cf_bias_scale * source_bias
            else:
                bias = bias + self.cf_bias_scale * source_bias
        return bias

    def forward(self,
                his_emb: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                src_mask: torch.Tensor = None,
                intent_assign=None,
                belief_confidence=None,
                source_gate=None,
                token_type=None):
        batch_size, seq_len, emb_size = his_emb.shape
        q = self._shape(self.q_proj(his_emb))
        k = self._shape(self.k_proj(his_emb))
        v = self._shape(self.v_proj(his_emb))

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(self.head_dim)

        bias = self._build_attention_bias(seq_len, intent_assign,
                                          belief_confidence, source_gate,
                                          token_type)
        if bias is not None:
            attn_logits = attn_logits + bias.unsqueeze(1)

        attn_mask = src_key_padding_mask[:, None, None, :].expand(
            -1, self.num_heads, seq_len, -1)
        if src_mask is not None:
            attn_mask = attn_mask | src_mask[:, None, :, :]

        all_masked = attn_mask.all(dim=-1, keepdim=True)
        attn_logits = attn_logits.masked_fill(attn_mask, -1e9)
        attn_logits = attn_logits.masked_fill(all_masked, 0)
        attn_weight = F.softmax(attn_logits, dim=-1)
        attn_weight = attn_weight.masked_fill(attn_mask | all_masked, 0)
        attn_weight = self.dropout(attn_weight)

        attn_output = torch.matmul(attn_weight, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, emb_size)
        return self.out_proj(attn_output)


class IntentSourceTransformerLayer(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, intent_bias_scale,
                 cf_bias_scale) -> None:
        super().__init__()
        self.self_attn = IntentSourceSelfAttention(emb_size, num_heads,
                                                   dropout,
                                                   intent_bias_scale,
                                                   cf_bias_scale)
        self.linear1 = nn.Linear(emb_size, emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self,
                his_emb: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                src_mask: torch.Tensor = None,
                intent_assign=None,
                belief_confidence=None,
                source_gate=None,
                token_type=None):
        attn_output = self.self_attn(his_emb, src_key_padding_mask, src_mask,
                                     intent_assign, belief_confidence,
                                     source_gate, token_type)
        his_emb = self.norm1(his_emb + self.dropout(attn_output))
        ff_output = self.linear2(self.dropout_ff(F.relu(self.linear1(his_emb))))
        his_emb = self.norm2(his_emb + self.dropout(ff_output))
        return his_emb


class Transformer(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers, dropout,
                 intent_bias_scale, cf_bias_scale) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            IntentSourceTransformerLayer(emb_size, num_heads, dropout,
                                         intent_bias_scale, cf_bias_scale)
            for _ in range(num_layers)
        ])

    def forward(self,
                his_emb: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                src_mask: torch.Tensor = None,
                intent_assign=None,
                belief_confidence=None,
                source_gate=None,
                token_type=None):
        his_encoded = his_emb
        for layer in self.layers:
            his_encoded = layer(his_encoded, src_key_padding_mask, src_mask,
                                intent_assign, belief_confidence, source_gate,
                                token_type)
        return his_encoded
