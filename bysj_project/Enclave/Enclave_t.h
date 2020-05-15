#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif
typedef struct MATRIX{
    float **p;
    int row;
    int col;
}MATRIX;

typedef struct safeLayerInfo{
    int input;
    int output;

    struct MATRIX weight;
    struct MATRIX safe_weight_rec;
    struct MATRIX safe_net;
    struct MATRIX safe_delta;
    struct MATRIX safe_weight_delta;
    struct MATRIX input_tensor;

}safeLayerInfo;
void Add_SafeLayer(unsigned int input, unsigned int output);
void FrontPropSafe(struct MATRIX input);
void BackPropSafe(struct MATRIX weight, struct MATRIX delta);
void Opt_safe(int type, float learn_rate, float decay_rate);
struct MATRIX get_safe_net(void);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
