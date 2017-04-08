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


// EigenTensorMap is same as `tensorflow::TTypes<T, NDIMS>::Tensor` 
// defined in "tensorflow/core/framework/tensor_types.h"
template<typename T, int NDIMS = 2>
using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>,
                                     Eigen::Aligned>;

template<typename T>
using EigenMatrixMap = Eigen::Map<
                        /*const*/ Eigen::Matrix<
                                  T,           /* scalar element type */
                                  Eigen::Dynamic,  /* num_rows is a run-time value */
                                  Eigen::Dynamic,  /* num_cols is a run-time value */
                                  Eigen::RowMajor  /* tensorflow::Tensor is always row-major */>
                              >;
template<typename T, int NDIMS = 2>
using EigenTensor = Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>;

template<typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;



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
      
    auto TMh_all = Mh_all.tensor<float, 3>(); // r x m x m
    auto TMt_all = Mt_all.tensor<float, 3>(); // r x m x m
    auto Trelations = relations.tensor<float, 3>(); // r x m x 1
    auto Tentitys = entitys.tensor<float, 3>();     // e x m x 1
    auto Tmask_h_all = mask_h_all.tensor<float, 3>();// r x m x m
    auto Tmask_t_all = mask_t_all.tensor<float, 3>();// r x m x m
    // auto Tlr = lr.scalar<float>()();
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

      EigenTensor<float> r_chip = Trelations.chip(rid, 0);
      EigenTensor<float> h_chip = Tentitys.chip(hid, 0);
      EigenTensor<float> t_chip = Tentitys.chip(tid, 0);
      EigenTensor<float> neg_h_chip = Tentitys.chip(n_hid, 0);
      EigenTensor<float> neg_t_chip = Tentitys.chip(n_tid, 0);
      EigenTensor<float> Mh_chip = TMh_all.chip(rid, 0);
      EigenTensor<float> Mt_chip = TMt_all.chip(rid, 0);
      EigenTensor<float> mask_h_chip = Tmask_h_all.chip(rid, 0);
      EigenTensor<float> mask_t_chip = Tmask_t_all.chip(rid, 0);


      auto r = EigenMatrixMap<float>(r_chip.data(), r_chip.dimension(0), r_chip.dimension(1));
      auto h = EigenMatrixMap<float>(h_chip.data(), h_chip.dimension(0), h_chip.dimension(1));
      auto t = EigenMatrixMap<float>(t_chip.data(), t_chip.dimension(0), t_chip.dimension(1));
      auto neg_h = EigenMatrixMap<float>(neg_h_chip.data(), neg_h_chip.dimension(0), neg_h_chip.dimension(1));
      auto neg_t = EigenMatrixMap<float>(neg_t_chip.data(), neg_t_chip.dimension(0), neg_t_chip.dimension(1));
      auto Mh = EigenMatrixMap<float>(Mh_chip.data(), Mh_chip.dimension(0), Mh_chip.dimension(1));
      auto Mt = EigenMatrixMap<float>(Mt_chip.data(), Mt_chip.dimension(0), Mt_chip.dimension(1));
      auto mask_h = EigenMatrixMap<float>(mask_h_chip.data(), mask_h_chip.dimension(0), mask_h_chip.dimension(1));
      auto mask_t = EigenMatrixMap<float>(mask_t_chip.data(), mask_t_chip.dimension(0), mask_t_chip.dimension(1));

      float lambda = 2 * lr.scalar<float>()();

      while(true){
        EigenMatrix<float> h_p = Mh * h;
        EigenMatrix<float> t_p = Mt * t;

        int nid = n_hid;
        EigenMatrix<float> neg_p = Mh * neg_h;
        if(!flag_head){
          nid = n_tid;         
          neg_p = Mt * neg_t;
        }
        float norm_h_p   = h_p.array().square().sum() - 1.;
        float norm_t_p   = t_p.array().square().sum() - 1.;
        float norm_neg_p = neg_p.array().square().sum() - 1.;
        float loss = (norm_h_p>0. ? norm_h_p : 0.) +
                     (norm_t_p>0. ? norm_t_p : 0.) +
                     (norm_neg_p>0. ? norm_neg_p : 0.);
        
        if (loss > 0.){
          // y = ||Wx||^2
          // dy/dx = 2W'(Wx)
          // dy/dW = 2(Wx)x'
          auto Mh_temp = Mh;
          auto Mt_temp = Mt;

          if (norm_h_p > 0.){
            Mh -= lambda * (h_p * h.transpose()).cwiseProduct(mask_h);
            h -= lambda * Mh_temp.transpose() * h_p;
           }
         
          
          if (norm_t_p > 0.){
            Mt -= lambda * (t_p * t.transpose()).cwiseProduct(mask_t);
            t -= lambda * Mt_temp.transpose() * t_p;
          }           

          if (norm_neg_p > 0.){
            if(flag_head){
              Mh -= lambda * (neg_p * neg_h.transpose()).cwiseProduct(mask_h);
              neg_h -= lambda * Mh_temp.transpose() * neg_p;
            }else{
              Mt -= lambda * (neg_p * neg_t.transpose()).cwiseProduct(mask_t);
              neg_t -= lambda * Mt_temp.transpose() * neg_p;
            }
           
          }
        }else{
          break;
        } // if loss > 0.
      }// while true
    
      // lock
      // entitys.tensor<float, 3>().chip(hid, 0) = EigenTensorMap<float>(h.data(), h_chip.dimensions());
      // entitys.tensor<float, 3>().chip(tid, 0) = EigenTensorMap<float>(t.data(), t_chip.dimensions());
      // if(flag_head){
      //     entitys.tensor<float, 3>().chip(n_hid, 0) = EigenTensorMap<float>(neg_h.data(), neg_h_chip.dimensions());
      //   }else{
      //     entitys.tensor<float, 3>().chip(n_tid, 0) = EigenTensorMap<float>(neg_t.data(), neg_t_chip.dimensions());
      // }
      // Mh_all.tensor<float, 3>().chip(rid, 0) = EigenTensorMap<float>(Mh.data(), Mh_chip.dimensions());
      // Mt_all.tensor<float, 3>().chip(rid, 0) = EigenTensorMap<float>(Mt.data(), Mt_chip.dimensions());
      // lock
      Tentitys.chip(hid, 0) = EigenTensorMap<float>(h.data(), h_chip.dimensions());
      Tentitys.chip(tid, 0) = EigenTensorMap<float>(t.data(), t_chip.dimensions());
      if(flag_head){
          Tentitys.chip(n_hid, 0) = EigenTensorMap<float>(neg_h.data(), neg_h_chip.dimensions());
        }else{
          Tentitys.chip(n_tid, 0) = EigenTensorMap<float>(neg_t.data(), neg_t_chip.dimensions());
      }
      TMh_all.chip(rid, 0) = EigenTensorMap<float>(Mh.data(), Mh_chip.dimensions());
      TMt_all.chip(rid, 0) = EigenTensorMap<float>(Mt.data(), Mt_chip.dimensions());
    }// for batch_size    
  }

 private:
  
};

REGISTER_KERNEL_BUILDER(Name("NormPrjctOp").Device(DEVICE_CPU), NormPrjctOp);







class NormPrjctOpV2 : public OpKernel {
 public:
  explicit NormPrjctOpV2(OpKernelConstruction* ctx) : OpKernel(ctx) {
    
  }

  // ~NormPrjctOpV2() { delete sampler_; }

  void Compute(OpKernelContext* ctx) override {

    const Tensor& t_mh = ctx->input(0); // b x m x m
    const Tensor& t_mt = ctx->input(1); // b x m x m
    const Tensor& t_h = ctx->input(2);  // b x m x 1
    const Tensor& t_t = ctx->input(3);  // b x m x 1
    const Tensor& t_neg_h = ctx->input(4);// b x m x 1
    const Tensor& t_neg_t = ctx->input(5);// b x m x 1
    const Tensor& t_flag_heads = ctx->input(6);// b
    const Tensor& t_mask_h = ctx->input(7); // b x m x m
    const Tensor& t_mask_t = ctx->input(8); // b x m x m
    const Tensor& lr = ctx->input(9);    // scalar
    
    auto Tmh = t_mh.tensor<float, 3>();
    auto Tmt = t_mt.tensor<float, 3>();
    auto Th = t_h.tensor<float, 3>();
    auto Tt = t_t.tensor<float, 3>();
    auto Tneg_h = t_neg_h.tensor<float, 3>();
    auto Tneg_t = t_neg_t.tensor<float, 3>();
    auto Tflag_heads = t_flag_heads.flat<bool>();
    auto Tmask_h = t_mask_h.tensor<float, 3>();
    auto Tmask_t = t_mask_t.tensor<float, 3>();

    const int64 batch_size = h.dim_size(0);
    float lambda = 2 * lr.scalar<float>()();
    

    for (int64 i = 0; i < batch_size; ++i){
      EigenTensor<float> Mh_chip = Tmh.chip(i, 0);
      EigenTensor<float> Mt_chip = Tmt.chip(i, 0);
      EigenTensor<float> h_chip = Th.chip(i, 0);
      EigenTensor<float> t_chip = Tt.chip(i, 0);
      EigenTensor<float> neg_h_chip = Tneg_h.chip(i, 0);
      EigenTensor<float> neg_t_chip = Tneg_t.chip(i, 0);
      bool flag_head = Tflag_heads(i);
      EigenTensor<float> mask_h_chip = Tmask_h.chip(i, 0);
      EigenTensor<float> mask_t_chip = Tmask_t.chip(i, 0);

      auto Mh = EigenMatrixMap<float>(Mh_chip.data(), Mh_chip.dimension(0), Mh_chip.dimension(1));
      auto Mt = EigenMatrixMap<float>(Mt_chip.data(), Mt_chip.dimension(0), Mt_chip.dimension(1));
      auto h = EigenMatrixMap<float>(h_chip.data(), h_chip.dimension(0), h_chip.dimension(1));
      auto t = EigenMatrixMap<float>(t_chip.data(), t_chip.dimension(0), t_chip.dimension(1));
      auto neg_h = EigenMatrixMap<float>(neg_h_chip.data(), neg_h_chip.dimension(0), neg_h_chip.dimension(1));
      auto neg_t = EigenMatrixMap<float>(neg_t_chip.data(), neg_t_chip.dimension(0), neg_t_chip.dimension(1));
      auto mask_h = EigenMatrixMap<float>(mask_h_chip.data(), mask_h_chip.dimension(0), mask_h_chip.dimension(1));
      auto mask_t = EigenMatrixMap<float>(mask_t_chip.data(), mask_t_chip.dimension(0), mask_t_chip.dimension(1));


      while(true){
        EigenMatrix<float> h_p = Mh * h;
        EigenMatrix<float> t_p = Mt * t;

        EigenMatrix<float> neg_p = Mh * neg_h;
        if(!flag_head){
          neg_p = Mt * neg_t;
        }
        float norm_h_p   = h_p.array().square().sum() - 1.;
        float norm_t_p   = t_p.array().square().sum() - 1.;
        float norm_neg_p = neg_p.array().square().sum() - 1.;
        float loss = (norm_h_p>0. ? norm_h_p : 0.) +
                     (norm_t_p>0. ? norm_t_p : 0.) +
                     (norm_neg_p>0. ? norm_neg_p : 0.);
        
        if (loss > 0.){
          // y = ||Wx||^2
          // dy/dx = 2W'(Wx)
          // dy/dW = 2(Wx)x'
          auto Mh_temp = Mh;
          auto Mt_temp = Mt;

          if (norm_h_p > 0.){
            Mh -= lambda * (h_p * h.transpose()).cwiseProduct(mask_h);
            h -= lambda * Mh_temp.transpose() * h_p;
           }
         
          
          if (norm_t_p > 0.){
            Mt -= lambda * (t_p * t.transpose()).cwiseProduct(mask_t);
            t -= lambda * Mt_temp.transpose() * t_p;
          }           

          if (norm_neg_p > 0.){
            if(flag_head){
              Mh -= lambda * (neg_p * neg_h.transpose()).cwiseProduct(mask_h);
              neg_h -= lambda * Mh_temp.transpose() * neg_p;
            }else{
              Mt -= lambda * (neg_p * neg_t.transpose()).cwiseProduct(mask_t);
              neg_t -= lambda * Mt_temp.transpose() * neg_p;
            }
           
          }
        }else{
          break;
        } // if loss > 0.
      }// while true
    
      // TODO: ??
      

      // lock
      // entitys.tensor<float, 3>().chip(hid, 0) = EigenTensorMap<float>(h.data(), h_chip.dimensions());
      // entitys.tensor<float, 3>().chip(tid, 0) = EigenTensorMap<float>(t.data(), t_chip.dimensions());
      // if(flag_head){
      //     entitys.tensor<float, 3>().chip(n_hid, 0) = EigenTensorMap<float>(neg_h.data(), neg_h_chip.dimensions());
      //   }else{
      //     entitys.tensor<float, 3>().chip(n_tid, 0) = EigenTensorMap<float>(neg_t.data(), neg_t_chip.dimensions());
      // }
      // Mh_all.tensor<float, 3>().chip(rid, 0) = EigenTensorMap<float>(Mh.data(), Mh_chip.dimensions());
      // Mt_all.tensor<float, 3>().chip(rid, 0) = EigenTensorMap<float>(Mt.data(), Mt_chip.dimensions());
      // lock
      Tentitys.chip(hid, 0) = EigenTensorMap<float>(h.data(), h_chip.dimensions());
      Tentitys.chip(tid, 0) = EigenTensorMap<float>(t.data(), t_chip.dimensions());
      if(flag_head){
          Tentitys.chip(n_hid, 0) = EigenTensorMap<float>(neg_h.data(), neg_h_chip.dimensions());
        }else{
          Tentitys.chip(n_tid, 0) = EigenTensorMap<float>(neg_t.data(), neg_t_chip.dimensions());
      }
      TMh_all.chip(rid, 0) = EigenTensorMap<float>(Mh.data(), Mh_chip.dimensions());
      TMt_all.chip(rid, 0) = EigenTensorMap<float>(Mt.data(), Mt_chip.dimensions());
    }// for batch_size    
  }

 private:
  
};

REGISTER_KERNEL_BUILDER(Name("NormPrjctOpV2").Device(DEVICE_CPU), NormPrjctOpV2);
}  // end namespace tensorflow
