# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""matcher"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.box_ops import box_cxcywh_to_xyxy
from src.box_ops import generalized_box_iou


def softmax(arr, axis=None):
    """softmax"""
    return np.exp(arr) / np.sum(np.exp(arr), axis=axis, keepdims=True)


class HungarianMatcher:
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def __call__(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        # We reshape to compute the cost matrices in a batch
        # out_prob [batch_size * num_queries, num_classes]
        out_prob = softmax(pred_logits.reshape(-1, pred_logits.shape[-1]), -1)
        # out_bbox [batch_size * num_queries, 4]
        out_bbox = pred_boxes.reshape(-1, pred_boxes.shape[-1])

        # Also concat the target labels and boxes
        tgt_ids = np.concatenate([v["labels"] for v in targets])
        tgt_bbox = np.concatenate([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = cdist(out_bbox, tgt_bbox, metric='minkowski', p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        final_cost_matrix = (self.cost_bbox * cost_bbox +
                             self.cost_class * cost_class +
                             self.cost_giou * cost_giou)
        final_cost_matrix = final_cost_matrix.reshape(bs, num_queries, -1)

        sizes = np.cumsum([len(v["boxes"]) for v in targets])
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(np.split(final_cost_matrix, sizes, -1)[:-1])
        ]
        return [(i, j) for i, j in indices]


def build_matcher(cfg):
    """build hungarian matcher"""
    return HungarianMatcher(cost_class=cfg.set_cost_class,
                            cost_bbox=cfg.set_cost_bbox,
                            cost_giou=cfg.set_cost_giou)
