#include "tensorflow/core/framework/op.h"

// ************************************************************************************************************
// NOTE: Useage of `tensorflow::Tensor` and `Eigen::Tensor`
// [tensorflow::Tensor](https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor)
// [Eigen::Tensor](https://bitbucket.org/eigen/eigen/src/677c9f1577810e869f4f09881cabc3e503a810c1/unsupported/Eigen/CXX11/src/Tensor/README.md)
// [Eigen::Tensor](http://eigen.tuxfamily.org/dox/unsupported/group__CXX11__Tensor__Module.html)
// [TensorChippingOp](https://eigen.tuxfamily.org/dox-devel/unsupported/TensorChipping_8h_source.html)
// [convert 2-D tensor to matrix](http://stackoverflow.com/questions/39475356/how-to-change-2d-eigentensor-to-eigenmatrix)
// [Eigen::Map] (http://eigen.tuxfamily.org/dox/classEigen_1_1Map.html)
// [Eigen::Matrix](http://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html)
// [Eigen::Matrix Arithmetic](http://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html)
// ************************************************************************************************************


namespace tensorflow {

REGISTER_OP("NormPrjctOp")
    .Input("transfer_matrix_h: Ref(float)")//Mh_all
    .Input("transfer_matrix_t: Ref(float)")//Mt_all
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
