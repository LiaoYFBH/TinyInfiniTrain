#include <cstddef>
#include <memory>
#include <cmath>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    int64_t num_elements = param->NumElements();
    float* grad_ptr = (float*)grad->DataPtr();
    float* param_ptr = (float*)param->DataPtr();
    float* m_ptr = (float*)m->DataPtr();
    float* v_ptr = (float*)v->DataPtr();

    #pragma omp parallel for
    for (int64_t i = 0; i < num_elements; ++i) {
        float g = grad_ptr[i];
        
        // m = beta1 * m + (1 - beta1) * grad
        m_ptr[i] = beta1 * m_ptr[i] + (1 - beta1) * g;
        
        // v = beta2 * v + (1 - beta2) * grad^2
        v_ptr[i] = beta2 * v_ptr[i] + (1 - beta2) * g * g;
        
        // bias correction
        float m_hat = m_ptr[i] / (1 - std::pow(beta1, t));
        float v_hat = v_ptr[i] / (1 - std::pow(beta2, t));
        
        // param update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
        param_ptr[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
