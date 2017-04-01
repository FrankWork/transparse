
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("NormPrjctOp")
    .Input("Mh_all: Ref(float)")
    .Input("Mt_all: Ref(float)")
    .Input("relations: Ref(float)")
    .Input("entitys: Ref(float)")
    .Input("mask_h_all: float")
    .Input("mask_t_all: float")
    .Input("lr: float")
    .Input("rids: int32")
    .Input("hids: int32")
    .Input("tids: int32")
    .Input("n_hids: int32")
    .Input("n_tids: int32")
    .Input("flag_heads: bool")
    .SetIsStateful()
    .Doc(R"doc(
Norm projected entities according to gradients.
)doc");

}  // end namespace tensorflow
