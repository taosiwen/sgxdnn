#ifndef ENCLAVE_U_H__
#define ENCLAVE_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "sgx_edger8r.h" /* for sgx_status_t etc. */


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

sgx_status_t Add_SafeLayer(sgx_enclave_id_t eid, unsigned int input, unsigned int output);
sgx_status_t FrontPropSafe(sgx_enclave_id_t eid, struct MATRIX input);
sgx_status_t BackPropSafe(sgx_enclave_id_t eid, struct MATRIX weight, struct MATRIX delta);
sgx_status_t Opt_safe(sgx_enclave_id_t eid, int type, float learn_rate, float decay_rate);
sgx_status_t get_safe_net(sgx_enclave_id_t eid, struct MATRIX* retval);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
