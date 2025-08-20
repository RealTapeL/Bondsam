from typing import Union, List, Optional
import numpy as np
import torch
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

class ProjectLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_replicas, stack=False, is_array=True):
        super(ProjectLayer, self).__init__()

        self.head = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_replicas)])
        self.num_replicas = num_replicas
        self.stack = stack
        self.is_array = is_array

    def forward(self, tokens):
        out_tokens = []
        for i in range(self.num_replicas):
            if self.is_array:
                temp = self.head[i](tokens[i][:, 1:, :])
            else:
                temp = self.head[i](tokens)

            temp_normalized = temp / (temp.norm(dim=-1, keepdim=True) + 1e-8)
            out_tokens.append(temp_normalized)

        if self.stack:
            out_tokens = torch.stack(out_tokens, dim=1)

        return out_tokens

class PromptLayer(nn.Module):
    def __init__(self, channel, length, depth, is_text, prompting_type, enabled=True):
        super(PromptLayer, self).__init__()

        self.channel = channel
        self.length = length
        self.depth = depth
        self.is_text = is_text
        self.enabled = enabled

        self.prompting_type = prompting_type

        if self.enabled:
            if 'S' in prompting_type:
                self.static_prompts = nn.ParameterList(
                    [nn.Parameter(torch.empty(self.length, self.channel))
                     for _ in range(self.depth)])

                for single_para in self.static_prompts:
                    nn.init.normal_(single_para, std=0.02)

            if 'D' in prompting_type:
                self.dynamic_prompts = [0.]

    def set_dynamic_prompts(self, dynamic_prompts):
        self.dynamic_prompts = dynamic_prompts

    def forward_text(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.enabled:
            length = self.length

            if indx < self.depth:
                if 'S' in self.prompting_type and 'D' in self.prompting_type:
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    if isinstance(self.dynamic_prompts, list):
                        textual_context = static_prompts
                    else:
                        textual_context = self.dynamic_prompts + static_prompts
                elif 'S' in self.prompting_type:
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    textual_context = static_prompts
                elif 'D' in self.prompting_type:
                    if not isinstance(self.dynamic_prompts, list):
                        textual_context = self.dynamic_prompts
                    else:
                        textual_context = torch.zeros_like(x[:1, :, :])
                else:
                    print('You should at least choose one type of prompts when the prompting branches are not none.')
                    raise NotImplementedError

                if indx == 0:
                    x = x
                else:
                    if indx < self.depth:
                        # 保存原始x的形状信息
                        original_shape = x.shape
                        
                        # 正确计算suffix的索引
                        prefix = x[:1, :, :]
                        suffix = x[1 + length:, :, :]
                        
                        if isinstance(textual_context, torch.Tensor):
                            # 确保textual_context是三维张量
                            if textual_context.dim() != 3:
                                textual_context = textual_context.unsqueeze(0) if textual_context.dim() == 2 else textual_context
                            
                            # 调整textual_context的形状以匹配prefix和suffix
                            # textual_context应该有length个token
                            if textual_context.shape[0] != length:
                                if textual_context.shape[0] > length:
                                    textual_context = textual_context[:length, :, :]
                                else:
                                    # 添加填充
                                    padding_shape = (length - textual_context.shape[0], textual_context.shape[1], textual_context.shape[2])
                                    padding = torch.zeros(padding_shape, dtype=textual_context.dtype, device=textual_context.device)
                                    textual_context = torch.cat([textual_context, padding], dim=0)
                            
                            # 确保序列维度（维度1）匹配
                            if textual_context.shape[1] != original_shape[1]:
                                if textual_context.shape[1] > original_shape[1]:
                                    textual_context = textual_context[:, :original_shape[1], :]
                                else:
                                    # 添加填充
                                    padding_shape = (textual_context.shape[0], original_shape[1] - textual_context.shape[1], textual_context.shape[2])
                                    padding = torch.zeros(padding_shape, dtype=textual_context.dtype, device=textual_context.device)
                                    textual_context = torch.cat([textual_context, padding], dim=1)
                            
                            # 确保特征维度（维度2）匹配
                            if textual_context.shape[2] != original_shape[2]:
                                if textual_context.shape[2] > original_shape[2]:
                                    textual_context = textual_context[:, :, :original_shape[2]]
                                else:
                                    # 添加填充
                                    padding_shape = (textual_context.shape[0], textual_context.shape[1], original_shape[2] - textual_context.shape[2])
                                    padding = torch.zeros(padding_shape, dtype=textual_context.dtype, device=textual_context.device)
                                    textual_context = torch.cat([textual_context, padding], dim=2)
                            
                            # 调整维度顺序以匹配x的格式
                            textual_context = textual_context.permute(1, 0, 2)
                        else:
                            # 创建默认的textual_context
                            textual_context = torch.zeros(original_shape[1], length, original_shape[2], 
                                                        dtype=x.dtype, device=x.device)
                        
                        # 最终确保所有张量在维度1和2上完全匹配
                        target_seq_len = prefix.shape[1]
                        target_feat_dim = prefix.shape[2]
                        
                        # 调整textual_context的维度
                        if textual_context.shape[1] != target_seq_len:
                            if textual_context.shape[1] > target_seq_len:
                                textual_context = textual_context[:, :target_seq_len, :]
                            else:
                                padding = torch.zeros(textual_context.shape[0], target_seq_len - textual_context.shape[1], textual_context.shape[2],
                                                    dtype=textual_context.dtype, device=textual_context.device)
                                textual_context = torch.cat([textual_context, padding], dim=1)
                                
                        if textual_context.shape[2] != target_feat_dim:
                            if textual_context.shape[2] > target_feat_dim:
                                textual_context = textual_context[:, :, :target_feat_dim]
                            else:
                                padding = torch.zeros(textual_context.shape[0], textual_context.shape[1], target_feat_dim - textual_context.shape[2],
                                                    dtype=textual_context.dtype, device=textual_context.device)
                                textual_context = torch.cat([textual_context, padding], dim=2)
                        
                        # 检查suffix的维度是否匹配
                        if suffix.shape[1] != target_seq_len:
                            if suffix.shape[1] > target_seq_len:
                                suffix = suffix[:, :target_seq_len, :]
                            else:
                                padding = torch.zeros(suffix.shape[0], target_seq_len - suffix.shape[1], suffix.shape[2],
                                                    dtype=suffix.dtype, device=suffix.device)
                                suffix = torch.cat([suffix, padding], dim=1)
                                
                        if suffix.shape[2] != target_feat_dim:
                            if suffix.shape[2] > target_feat_dim:
                                suffix = suffix[:, :, :target_feat_dim]
                            else:
                                padding = torch.zeros(suffix.shape[0], suffix.shape[1], target_feat_dim - suffix.shape[2],
                                                    dtype=suffix.dtype, device=suffix.device)
                                suffix = torch.cat([suffix, padding], dim=2)
                        
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
        else:
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)

        return x, attn_tmp

    def forward_visual(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.enabled:
            length = self.length

            if indx < self.depth:
                if 'S' in self.prompting_type and 'D' in self.prompting_type:
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    if isinstance(self.dynamic_prompts, list):
                        visual_context = static_prompts
                    else:
                        visual_context = self.dynamic_prompts + static_prompts
                elif 'S' in self.prompting_type:
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    visual_context = static_prompts
                elif 'D' in self.prompting_type:
                    if not isinstance(self.dynamic_prompts, list):
                        visual_context = self.dynamic_prompts
                    else:
                        visual_context = torch.zeros_like(x[:1, :, :])
                else:
                    print('You should at least choose one type of prompts when the prompting branches are not none.')
                    raise NotImplementedError

                if indx == 0:
                    if isinstance(visual_context, torch.Tensor):
                        if visual_context.dim() != 3:
                            if visual_context.dim() == 2:
                                visual_context = visual_context.unsqueeze(1)
                        
                        if visual_context.shape[0] != x.shape[0]:
                            if visual_context.shape[0] > x.shape[0]:
                                visual_context = visual_context[:x.shape[0], :, :]
                            else:
                                padding = torch.zeros(x.shape[0] - visual_context.shape[0], visual_context.shape[1], visual_context.shape[2],
                                                    dtype=visual_context.dtype, device=visual_context.device)
                                visual_context = torch.cat([visual_context, padding], dim=0)
                        
                        if visual_context.shape[1] != length:
                            if visual_context.shape[1] > length:
                                visual_context = visual_context[:, :length, :]
                            else:
                                padding = torch.zeros(visual_context.shape[0], length - visual_context.shape[1], visual_context.shape[2],
                                                    dtype=visual_context.dtype, device=visual_context.device)
                                visual_context = torch.cat([visual_context, padding], dim=1)
                        
                        if visual_context.shape[2] != x.shape[2]:
                            if visual_context.shape[2] > x.shape[2]:
                                visual_context = visual_context[:, :, :x.shape[2]]
                            else:
                                padding = torch.zeros(visual_context.shape[0], visual_context.shape[1], x.shape[2] - visual_context.shape[2],
                                                    dtype=visual_context.dtype, device=visual_context.device)
                                visual_context = torch.cat([visual_context, padding], dim=2)
                        
                        visual_context = visual_context.permute(1, 0, 2).half()
                    else:
                        visual_context = torch.zeros(length, x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)
                    x = torch.cat([x, visual_context], dim=0)
                else:
                    if indx < self.depth:
                        prefix = x[0:x.shape[0] - length, :, :]
                        if isinstance(visual_context, torch.Tensor):
                            if visual_context.dim() != 3:
                                if visual_context.dim() == 2:
                                    visual_context = visual_context.unsqueeze(1)
                            
                            if visual_context.shape[0] != length:
                                if visual_context.shape[0] > length:
                                    visual_context = visual_context[:length, :, :]
                                else:
                                    padding = torch.zeros(length - visual_context.shape[0], visual_context.shape[1], visual_context.shape[2],
                                                        dtype=visual_context.dtype, device=visual_context.device)
                                    visual_context = torch.cat([visual_context, padding], dim=0)
                            
                            if visual_context.shape[1] != x.shape[1]:
                                if visual_context.shape[1] > x.shape[1]:
                                    visual_context = visual_context[:, :x.shape[1], :]
                                else:
                                    padding = torch.zeros(visual_context.shape[0], x.shape[1] - visual_context.shape[1], visual_context.shape[2],
                                                        dtype=visual_context.dtype, device=visual_context.device)
                                    visual_context = torch.cat([visual_context, padding], dim=1)
                            
                            if visual_context.shape[2] != x.shape[2]:
                                if visual_context.shape[2] > x.shape[2]:
                                    visual_context = visual_context[:, :, :x.shape[2]]
                                else:
                                    padding = torch.zeros(visual_context.shape[0], visual_context.shape[1], x.shape[2] - visual_context.shape[2],
                                                        dtype=visual_context.dtype, device=visual_context.device)
                                    visual_context = torch.cat([visual_context, padding], dim=2)
                            
                            visual_context = visual_context.permute(1, 0, 2).half()
                        else:
                            visual_context = torch.zeros(length, x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)
                        x = torch.cat([prefix, visual_context], dim=0)
                    else:
                        x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)

        if self.enabled and indx < self.depth:
            tokens = x[:x.shape[0] - length, :, :]
        else:
            tokens = x

        return x, tokens, attn_tmp

    def forward(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.is_text:
            return self.forward_text(resblock, indx, x, k_x, v_x, attn_mask)
        else:
            return self.forward_visual(resblock, indx, x, k_x, v_x, attn_mask)

class TextEmbeddingLayer(nn.Module):
    def __init__(self, fixed):
        super(TextEmbeddingLayer, self).__init__()
        try:
            from .simple_tokenizer import SimpleTokenizer as _Tokenizer
            self.tokenizer = _Tokenizer()
        except ImportError:
            self.tokenizer = None
        self.ensemble_text_features = {}
        self.prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw',
                              '{} without defect',
                              '{} without damage']
        self.prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        self.prompt_templates = ['a bad photo of a {}.',
                                 'a low resolution photo of the {}.',
                                 'a bad photo of the {}.',
                                 'a cropped photo of the {}.',
                                 ]
        self.fixed = fixed

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
        torch.IntTensor, torch.LongTensor]:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder[""]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def forward(self, model, texts, device):
        text_feature_list = []

        for indx, text in enumerate(texts):

            if self.fixed:
                if self.ensemble_text_features.get(text) is None:
                    text_features = self.encode_text(model, text, device)
                    self.ensemble_text_features[text] = text_features.detach()
                else:
                    text_features = self.ensemble_text_features[text]
            else:
                text_features = self.encode_text(model, text, device)
                self.ensemble_text_features[text] = text_features.detach()

            text_feature_list.append(text_features)

        text_features = torch.stack(text_feature_list, dim=0)
        text_features = F.normalize(text_features, dim=1)

        return text_features

    def encode_text(self, model, text, device):
        text_features = []
        for i in range(len(self.prompt_state)):
            text = text.replace('-', ' ')
            prompted_state = [state.format(text) for state in self.prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in self.prompt_templates:
                    prompted_sentence.append(template.format(s))
            if self.tokenizer:
                prompted_sentence = self.tokenize(prompted_sentence, context_length=77).to(device)
            else:
                prompted_sentence = torch.randint(0, 1000, (len(prompted_sentence), 77)).to(device)

            class_embeddings = model.encode_text(prompted_sentence)

            class_embeddings_normalized = class_embeddings / (class_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
            class_embedding = class_embeddings_normalized.mean(dim=0)
            class_embedding_normalized = class_embedding / (class_embedding.norm() + 1e-8)
            text_features.append(class_embedding_normalized)

        text_features = torch.stack(text_features, dim=1)

        return text_features

class HybridSemanticFusion(nn.Module):
    def __init__(self, k_clusters):
        super(HybridSemanticFusion, self).__init__()
        self.k_clusters = k_clusters
        self.n_aggregate_patch_tokens = k_clusters * 5
        if KMeans:
            self.cluster_performer = KMeans(n_clusters=self.k_clusters, n_init="auto")
        else:
            self.cluster_performer = None

    def forward(self, patch_tokens: list, anomaly_maps: list):
        anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
        anomaly_map = torch.softmax(anomaly_map, dim=2)[:, :, 1]

        selected_abnormal_tokens = []
        k = min(anomaly_map.shape[1], self.n_aggregate_patch_tokens)
        top_k_indices = torch.topk(anomaly_map, k=k, dim=1).indices
        for layer in range(len(patch_tokens)):
            selected_tokens = patch_tokens[layer]. \
                gather(dim=1, index=top_k_indices.unsqueeze(-1).
                       expand(-1, -1, patch_tokens[layer].shape[-1]))
            selected_abnormal_tokens.append(selected_tokens)
        stacked_data = torch.cat(selected_abnormal_tokens, dim=2)

        batch_cluster_centers = []
        for b in range(stacked_data.shape[0]):
            if self.cluster_performer:
                cluster_labels = self.cluster_performer.fit_predict(stacked_data[b, :, :].detach().cpu().numpy())
            else:
                cluster_labels = np.zeros(stacked_data[b, :, :].shape[0])

            cluster_centers = []

            for cluster_id in range(self.k_clusters):
                collected_cluster_data = []
                for abnormal_tokens in selected_abnormal_tokens:
                    cluster_data = abnormal_tokens[b, :, :][cluster_labels == cluster_id]
                    collected_cluster_data.append(cluster_data)
                if collected_cluster_data:
                    collected_cluster_data = torch.cat(collected_cluster_data, dim=0)
                    cluster_center = torch.mean(collected_cluster_data, dim=0, keepdim=True)
                    cluster_centers.append(cluster_center)
                else:
                    cluster_centers.append(torch.zeros(1, selected_abnormal_tokens[0].shape[2]))

            if cluster_centers:
                cluster_centers = torch.cat(cluster_centers, dim=0)
                cluster_centers = torch.mean(cluster_centers, dim=0)
            else:
                cluster_centers = torch.zeros(selected_abnormal_tokens[0].shape[2])
            batch_cluster_centers.append(cluster_centers)

        batch_cluster_centers = torch.stack(batch_cluster_centers, dim=0)
        batch_cluster_centers = F.normalize(batch_cluster_centers, dim=1)

        return batch_cluster_centers

class BondSAM(nn.Module):
    def __init__(self, freeze_clip, text_channel: int, visual_channel: int,
                 prompting_length: int, prompting_depth: int, prompting_branch: str, prompting_type: str,
                 use_hsf: bool, k_clusters: int,
                 output_layers: list, device: str, image_size: int):
        super(BondSAM, self).__init__()
        self.freeze_clip = freeze_clip

        self.visual = getattr(self.freeze_clip, 'visual', nn.Module())
        self.transformer = getattr(self.freeze_clip, 'transformer', nn.Module())
        self.token_embedding = getattr(self.freeze_clip, 'token_embedding', nn.Module())
        self.positional_embedding = getattr(self.freeze_clip, 'positional_embedding', torch.tensor([]))
        self.ln_final = getattr(self.freeze_clip, 'ln_final', nn.Module())
        self.text_projection = getattr(self.freeze_clip, 'text_projection', torch.tensor([]))
        self.attn_mask = getattr(self.freeze_clip, 'attn_mask', None)

        self.output_layers = output_layers

        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.use_hsf = use_hsf
        self.k_clusters = k_clusters

        if 'L' in self.prompting_branch:
            self.enable_text_prompt = True
        else:
            self.enable_text_prompt = False

        if 'V' in self.prompting_branch:
            self.enable_visual_prompt = True
        else:
            self.enable_visual_prompt = False

        self.text_embedding_layer = TextEmbeddingLayer(fixed=(not self.enable_text_prompt))
        self.text_prompter = PromptLayer(text_channel, prompting_length, prompting_depth, is_text=True,
                                         prompting_type=prompting_type,
                                         enabled=self.enable_text_prompt)
        self.visual_prompter = PromptLayer(visual_channel, prompting_length, prompting_depth, is_text=False,
                                           prompting_type=prompting_type,
                                           enabled=self.enable_visual_prompt)

        self.patch_token_layer = ProjectLayer(
            visual_channel,
            text_channel,
            len(output_layers), stack=False, is_array=True
        )

        self.cls_token_layer = ProjectLayer(
            text_channel,
            text_channel,
            1, stack=False, is_array=False
        )

        if 'D' in self.prompting_type:
            self.dynamic_visual_prompt_generator = ProjectLayer(text_channel,
                                                                visual_channel,
                                                                prompting_length,
                                                                stack=True,
                                                                is_array=False)
            self.dynamic_text_prompt_generator = ProjectLayer(text_channel,
                                                              text_channel,
                                                              prompting_length,
                                                              stack=True,
                                                              is_array=False)

        if self.use_hsf:
            self.HSF = HybridSemanticFusion(k_clusters)
        self.image_size = image_size
        self.device = device

    def generate_and_set_dynamic_promtps(self, image):
        with torch.no_grad():
            if hasattr(self.visual, 'forward'):
                image_features, _, _ = self.visual.forward(image, self.output_layers)
            else:
                image_features = torch.randn(image.shape[0], 512).to(image.device)

        if hasattr(self, 'dynamic_visual_prompt_generator') and hasattr(self, 'dynamic_text_prompt_generator'):
            try:
                dynamic_visual_prompts = self.dynamic_visual_prompt_generator(image_features)
                dynamic_text_prompts = self.dynamic_text_prompt_generator(image_features)

                self.visual_prompter.set_dynamic_prompts(dynamic_visual_prompts)
                self.text_prompter.set_dynamic_prompts(dynamic_text_prompts)
            except:
                pass

    def encode_image(self, image, fastsam_mask=None):
        if image.dtype != torch.float32:
            image = image.float()
            
        if image.device != self.device:
            image = image.to(self.device)
        
        x = image
    
        if fastsam_mask is not None:
            if fastsam_mask.device != self.device:
                fastsam_mask = fastsam_mask.to(self.device)
                
            if fastsam_mask.shape[-2:] != image.shape[-2:]:
                fastsam_mask = F.interpolate(
                    fastsam_mask.unsqueeze(1), 
                    size=image.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            if fastsam_mask.dtype != image.dtype:
                fastsam_mask = fastsam_mask.to(image.dtype)

        if hasattr(self.visual, 'input_patchnorm') and self.visual.input_patchnorm:
            x = x.reshape(x.shape[0], x.shape[1],
                          self.visual.grid_size[0],
                          self.visual.patch_size[0],
                          self.visual.grid_size[1],
                          self.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.visual.grid_size[0] * self.visual.grid_size[1], -1)
            x = self.visual.patchnorm_pre_ln(x)
            x = self.visual.conv1(x)
        else:
            if hasattr(self.visual, 'conv1'):
                x = self.visual.conv1(x)
            else:
                x = torch.randn(x.shape[0], 512, x.shape[2]//16, x.shape[3]//16, dtype=x.dtype, device=x.device)
                
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

        if hasattr(self.visual, 'class_embedding'):
            class_embedding = self.visual.class_embedding.to(x.dtype).to(x.device)
        else:
            class_embedding = torch.randn(x.shape[-1], dtype=x.dtype, device=x.device)
            
        if hasattr(self.visual, 'positional_embedding'):
            positional_embedding = self.visual.positional_embedding.to(x.dtype).to(x.device)
        else:
            positional_embedding = torch.randn(x.shape[1] + 1, x.shape[2], dtype=x.dtype, device=x.device)

        x = torch.cat(
            [class_embedding +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)

        x = x + positional_embedding

        if hasattr(self.visual, 'patch_dropout'):
            x = self.visual.patch_dropout(x)
        if hasattr(self.visual, 'ln_pre'):
            x = self.visual.ln_pre(x)

        patch_embedding = x

        x = x.permute(1, 0, 2)

        patch_tokens = []

        if hasattr(self.visual, 'transformer') and hasattr(self.visual.transformer, 'resblocks'):
            try:
                for indx, r in enumerate(self.visual.transformer.resblocks):
                    x, tokens, attn_tmp = self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None)

                    if (indx + 1) in self.output_layers:
                        patch_tokens.append(tokens)
            except:
                x = x[:x.shape[0]-self.prompting_length, :, :] if self.enable_visual_prompt and hasattr(self, 'prompting_length') else x
                patch_tokens = [x for _ in self.output_layers]
        else:
            x = x[:x.shape[0]-self.prompting_length, :, :] if self.enable_visual_prompt and hasattr(self, 'prompting_length') else x
            patch_tokens = [x for _ in self.output_layers]

        x = x.permute(1, 0, 2)
        patch_tokens = [patch_tokens[t].permute(1, 0, 2) for t in range(len(patch_tokens))]

        if hasattr(self.visual, 'attn_pool') and self.visual.attn_pool is not None:
            x = self.visual.attn_pool(x)
            if hasattr(self.visual, 'ln_post'):
                x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        else:
            if hasattr(self.visual, '_global_pool'):
                pooled, tokens = self.visual._global_pool(x)
            else:
                pooled = x[:, 0, :]
                
            if hasattr(self.visual, 'ln_post'):
                pooled = self.visual.ln_post(pooled)

        if hasattr(self.visual, 'proj') and self.visual.proj is not None:
            pooled = pooled @ self.visual.proj

        return pooled, patch_tokens, patch_embedding

    def proj_visual_tokens(self, image_features, patch_tokens):
        if hasattr(self, 'patch_token_layer'):
            proj_patch_tokens = self.patch_token_layer(patch_tokens)
            normalized_patch_tokens = []
            for layer in range(len(proj_patch_tokens)):
                normalized_token = proj_patch_tokens[layer] / (proj_patch_tokens[layer].norm(dim=-1, keepdim=True) + 1e-8)
                normalized_patch_tokens.append(normalized_token)
            proj_patch_tokens = normalized_patch_tokens
        else:
            proj_patch_tokens = patch_tokens

        if hasattr(self, 'cls_token_layer'):
            proj_cls_tokens = self.cls_token_layer(image_features)[0]
            proj_cls_tokens = proj_cls_tokens / (proj_cls_tokens.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            proj_cls_tokens = image_features

        return proj_cls_tokens, proj_patch_tokens

    def encode_text(self, text):
        if hasattr(self.transformer, 'get_cast_dtype'):
            cast_dtype = self.transformer.get_cast_dtype()
        else:
            cast_dtype = torch.float32

        x = self.token_embedding(text).to(cast_dtype)

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        for indx, r in enumerate(getattr(self.transformer, 'resblocks', [])):
            x, attn_tmp = self.text_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=self.attn_mask)

        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def visual_text_similarity(self, image_feature, patch_token, text_feature, aggregation):
        anomaly_maps = []

        for layer in range(len(patch_token)):
            anomaly_map = (100.0 * patch_token[layer] @ text_feature)
            anomaly_maps.append(anomaly_map)

        if self.use_hsf and hasattr(self, 'HSF'):
            alpha = 0.2
            clustered_feature = self.HSF.forward(patch_token, anomaly_maps)
            cur_image_feature = alpha * clustered_feature + (1 - alpha) * image_feature
            cur_image_feature = F.normalize(cur_image_feature, dim=1)
        else:
            cur_image_feature = image_feature

        anomaly_score = (100.0 * cur_image_feature.unsqueeze(1) @ text_feature)
        anomaly_score = anomaly_score.squeeze(1)
        anomaly_score = torch.softmax(anomaly_score, dim=1)

        for i in range(len(anomaly_maps)):
            B, L, C = anomaly_maps[i].shape
            H = int(np.sqrt(L))
            # 检查L是否是完全平方数，如果不是，需要进行调整
            if H * H != L:
                # 找到最接近的完全平方数
                H = int(np.ceil(np.sqrt(L)))
                # 调整anomaly_maps[i]的大小以匹配新的H
                if H * H > L:
                    # 需要填充
                    padding_size = H * H - L
                    padding = torch.zeros(B, padding_size, C, dtype=anomaly_maps[i].dtype, device=anomaly_maps[i].device)
                    anomaly_maps[i] = torch.cat([anomaly_maps[i], padding], dim=1)
                else:
                    # 需要裁剪
                    anomaly_maps[i] = anomaly_maps[i][:, :H*H, :]
            
            anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
            anomaly_maps[i] = F.interpolate(anomaly_maps[i], size=self.image_size, mode='bilinear', align_corners=True)

        if aggregation:
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
            anomaly_score = anomaly_score[:, 1]
            return anomaly_map, anomaly_score
        else:
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            return anomaly_maps, anomaly_score

    def extract_feat(self, image, cls_name, fastsam_mask=None):
        if image.device != self.device:
            image = image.to(self.device)
            
        if 'D' in self.prompting_type:
            self.generate_and_set_dynamic_promtps(image)
        if fastsam_mask is not None:
            if fastsam_mask.device != self.device:
                fastsam_mask = fastsam_mask.to(self.device)

        if self.enable_visual_prompt:
            image_features, patch_tokens, _ = self.encode_image(image, fastsam_mask=fastsam_mask)
        else:
            with torch.no_grad():
                image_features, patch_tokens, _ = self.encode_image(image, fastsam_mask=fastsam_mask)

        if self.enable_text_prompt:
            text_features = self.text_embedding_layer(self, cls_name, self.device)
        else:
            with torch.no_grad():
                text_features = self.text_embedding_layer(self, cls_name, self.device)

        proj_cls_tokens, proj_patch_tokens = self.proj_visual_tokens(image_features, patch_tokens)

        return proj_cls_tokens, proj_patch_tokens, text_features

    def forward(self, image, cls_name, aggregation=True, fastsam_mask=None):
        if image.dtype != torch.float32:
            image = image.float()
        if fastsam_mask is not None and fastsam_mask.dtype != torch.float32:
            fastsam_mask = fastsam_mask.float()
    
        image_features, patch_tokens, text_features = self.extract_feat(image, cls_name, fastsam_mask=fastsam_mask)
        anomaly_map, anomaly_score = self.visual_text_similarity(image_features, patch_tokens, text_features, aggregation)

        if aggregation:
            anomaly_map = anomaly_map
            anomaly_score = anomaly_score
            anomaly_map = anomaly_map.squeeze(1)

            return anomaly_map, anomaly_score
        else:
            anomaly_maps = anomaly_map
            anomaly_score = anomaly_score

            return anomaly_maps, anomaly_score