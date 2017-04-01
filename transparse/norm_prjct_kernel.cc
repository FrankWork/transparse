#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/lib/core/stringpiece.h"
// #include "tensorflow/core/lib/gtl/map_util.h"
// #include "tensorflow/core/lib/random/distribution_sampler.h"
// #include "tensorflow/core/lib/random/philox_random.h"
// #include "tensorflow/core/lib/random/simple_philox.h"
// #include "tensorflow/core/lib/strings/str_util.h"
// #include "tensorflow/core/platform/thread_annotations.h"
// #include "tensorflow/core/util/guarded_philox_random.h"

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


// EigenTensorMap is same as `tensorflow::TTypes<T, NDIMS>::Tensor` 
// defined in "tensorflow/core/framework/tensor_types.h"
template<typename T, int NDIMS = 2>
using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>,
                                     Eigen::Aligned>;

template<typename T>
using EigenMatrixMap = Eigen::Map<
                        const Eigen::Matrix<
                                  T,           /* scalar element type */
                                  Eigen::Dynamic,  /* num_rows is a run-time value */
                                  Eigen::Dynamic,  /* num_cols is a run-time value */
                                  Eigen::RowMajor  /* tensorflow::Tensor is always row-major */>
                              >;
template<typename T, int NDIMS = 2>
using EigenTensor = Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>;




namespace tensorflow {

class NormPrjctOp : public OpKernel {
 public:
  explicit NormPrjctOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    
  }

  // ~NormPrjctOp() { delete sampler_; }

  void Compute(OpKernelContext* ctx) override {
    Tensor Mh_all = ctx->mutable_input(0, false);
    // OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(Mh_all),
    //             errors::InvalidArgument("Must be a matrix or higher"));
    Tensor Mt_all = ctx->mutable_input(1, false);
    // OP_REQUIRES(ctx, Mh_all.shape() == Mt_all.shape(),
    //             errors::InvalidArgument("Mh_all.shape() == Mt_all.shape()"));
    Tensor relations = ctx->mutable_input(2, false);
    // OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(relations),
    //             errors::InvalidArgument("Must be a matrix or higher"));
    Tensor entitys = ctx->mutable_input(3, false);
    // OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(entitys),
    //             errors::InvalidArgument("Must be a matrix or higher"));
    const Tensor& mask_h_all = ctx->input(4);
    // OP_REQUIRES(ctx, mask_h_all.shape() == Mh_all.shape(),
    //             erros::InvalidArgument("mask_h_all.shape() == Mh_all.shape()"));
    const Tensor& mask_t_all = ctx->input(5);
    // OP_REQUIRES(ctx, mask_h_all.shape() == mask_t_all.shape(),
    //             erros::InvalidArgument("mask_h_all.shape() == mask_t_all.shape()"));
    const Tensor& lr = ctx->input(6);
    // OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
    //             erros::InvalidArgument("Must be a scalar"));
    const Tensor& rids = ctx->input(7);
    // OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(rids.shape()),
    //             erros::InvalidArgument("Must be a matrix or higher"));
    const Tensor& hids = ctx->input(8);
    // OP_REQUIRES(ctx, rids.shape() == hids.shape(),
    //             erros::InvalidArgument("rids.shape == hids.shape"))
    const Tensor& tids = ctx->input(9);
    // OP_REQUIRES(ctx, rids.shape() == tids.shape(),
    //             erros::InvalidArgument("rids.shape == tids.shape"))
    const Tensor& n_hids = ctx->input(10);
    // OP_REQUIRES(ctx, rids.shape() == n_hids.shape(),
    //             erros::InvalidArgument("rids.shape == n_hids.shape"))
    const Tensor& n_tids = ctx->input(11);
    // OP_REQUIRES(ctx, rids.shape() == n_tids.shape(),
    //             erros::InvalidArgument("rids.shape == n_tids.shape"))
    const Tensor& flag_heads = ctx->input(12);
    // OP_REQUIRES(ctx, rids.shape() == flag_heads.shape(),
    //             erros::InvalidArgument("rids.shape == flag_heads.shape"))


    auto TMh_all = Mh_all.tensor<float>(); // r x m x m
    auto TMt_all = Mt_all.tensor<float>(); // r x m x m
    auto Trelations = relations.tensor<float>(); // r x m x 1
    auto Tentitys = entitys.tensor<float>();     // e x m x 1
    auto Tmask_h_all = mask_h_all.tensor<float>();// r x m x m
    auto Tmask_t_all = mask_t_all.tensor<float>();// r x m x m
    auto Tlr = lr.scalar<float>()();
    auto Trids = rids.flat<int32>(); // b 
    auto Thids = hids.flat<int32>(); // b 
    auto Ttids = tids.flat<int32>(); // b 
    auto Tn_hids = n_hids.flat<int32>(); // b 
    auto Tn_tids = n_tids.flat<int32>(); // b 
    auto Tflag_heads = flag_heads.flat<bool>(); // b 

    const int64 batch_size = rids.dim_size(0);

    for (int64 i = 0; i < batch_size; ++i){
      int rid = Trids(i);
      int hid = Thids(i);
      int tid = Ttids(i);
      int n_hid = Tn_hids(i);
      int n_tid = Tn_tids(i);
      bool flag_head = Tflag_heads(i);

      auto r = EigenMatrixMap<float>(Trelations.chip(rid, 0));
      auto h = EigenMatrixMap<float>(Tentitys.chip(hid, 0));
      auto t = EigenMatrixMap<float>(Tentitys.chip(tid, 0));
      auto neg_h = EigenMatrixMap<float>(Tentitys.chip(n_hid, 0));
      auto neg_t = EigenMatrixMap<float>(Tentitys.chip(n_tid, 0));
      auto Mh = EigenMatrixMap<float>(TMh_all.chip(rid, 0));
      auto Mt = EigenMatrixMap<float>(TMt_all.chip(rid, 0));
      auto mask_h = EigenMatrixMap<float>(Tmask_h_all.chip(rid, 0));
      auto mask_t = EigenMatrixMap<float>(Tmask_t_all.chip(rid, 0));

      while(true){
        auto h_p = Mh * h;
        auto t_p = Mt * t;

        auto nid = n_hid;
        auto neg_p = Mh * neg_h;
        if(!flag_head){
          nid = n_tid;        
          neg_p = Mt * neg_t;
        }
        float norm_h_p   = h_p.pow(2).sum()() - 1.;
        float norm_t_p   = t_p.pow(2).sum()() - 1.;
        float norm_neg_p = neg_p.pow(2).sum()() - 1.;
        float loss = (norm_h_p>0. ? norm_h_p : 0.) +
                     (norm_t_p>0. ? norm_t_p : 0.) +
                     (norm_neg_p>0. ? norm_neg_p : 0.);

        if (loss > 0.){
          // y = Wx
          // dy/dx = 2(Wx)'W
          // dy/dW = 2(Wx)x'
          if (norm_h_p > 0.){
            Mh -= lr * 2 * (h_p * h.transpose()).cwiseProduct(mask_h);
            h -= lr * 2 * h_p.transpose() * Mh;
          }
          if (norm_t_p > 0.){
            Mt -= lr * 2 * (t_p * t.transpose()).cwiseProduct(mask_t);
            t -= lr * 2 * t_p.transpose() * Mt;
          }

          if (norm_neg_p > 0.){
            if(flag_head){
              Mh -= lr * 2 * (neg_p * neg_h.transpose()).cwiseProduct(mask_h);
              neg_h -= lr * 2 * neg_p.transpose() * Mh;
            }else{
              Mt -= lr * 2 * (neg_p * neg_t.transpose()).cwiseProduct(mask_t);
              neg_t -= lr * 2 * neg_p.transpose() * Mt;
            }
           
          }
          
        }else{
          break;
        } // if loss > 0.
      }// while true
    
	  Tentitys.chip(hid, 0) = EigenTensorMap<float>(h.data(), Tentitys.chip(hid, 0).dimensions());
	  Tentitys.chip(tid, 0) = EigenTensorMap<float>(t.data(), Tentitys.chip(tid, 0).dimensions());
	  if(flag_head()){
	    	Tentitys.chip(n_hid, 0) = EigenTensorMap<float>(neg_h.data(), Tentitys.chip(n_hid, 0).dimensions());
      }else{
        Tentitys.chip(n_tid, 0) = EigenTensorMap<float>(neg_t.data(), Tentitys.chip(n_tid, 0).dimensions());
    }
	  TMh_all.chip(rid, 0) = EigenTensorMap<float>(Mh.data(), TMh_all.chip(rid, 0).dimensions());
	  TMt_all.chip(rid, 0) = EigenTensorMap<float>(Mt.data(), TMt_all.chip(rid, 0).dimensions());
    }// for batch_size    
  }

 private:
  
};

REGISTER_KERNEL_BUILDER(Name("NormPrjctOp").Device(DEVICE_CPU), NormPrjctOp);

}  // end namespace tensorflow
