#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_Add_SafeLayer_t {
	unsigned int ms_input;
	unsigned int ms_output;
} ms_Add_SafeLayer_t;

typedef struct ms_FrontPropSafe_t {
	struct MATRIX ms_input;
} ms_FrontPropSafe_t;

typedef struct ms_BackPropSafe_t {
	struct MATRIX ms_weight;
	struct MATRIX ms_delta;
} ms_BackPropSafe_t;

typedef struct ms_Opt_safe_t {
	int ms_type;
	float ms_learn_rate;
	float ms_decay_rate;
} ms_Opt_safe_t;

typedef struct ms_get_safe_net_t {
	struct MATRIX ms_retval;
} ms_get_safe_net_t;

static const struct {
	size_t nr_ocall;
	void * table[1];
} ocall_table_Enclave = {
	0,
	{ NULL },
};
sgx_status_t Add_SafeLayer(sgx_enclave_id_t eid, unsigned int input, unsigned int output)
{
	sgx_status_t status;
	ms_Add_SafeLayer_t ms;
	ms.ms_input = input;
	ms.ms_output = output;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t FrontPropSafe(sgx_enclave_id_t eid, struct MATRIX input)
{
	sgx_status_t status;
	ms_FrontPropSafe_t ms;
	ms.ms_input = input;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t BackPropSafe(sgx_enclave_id_t eid, struct MATRIX weight, struct MATRIX delta)
{
	sgx_status_t status;
	ms_BackPropSafe_t ms;
	ms.ms_weight = weight;
	ms.ms_delta = delta;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t Opt_safe(sgx_enclave_id_t eid, int type, float learn_rate, float decay_rate)
{
	sgx_status_t status;
	ms_Opt_safe_t ms;
	ms.ms_type = type;
	ms.ms_learn_rate = learn_rate;
	ms.ms_decay_rate = decay_rate;
	status = sgx_ecall(eid, 3, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t get_safe_net(sgx_enclave_id_t eid, struct MATRIX* retval)
{
	sgx_status_t status;
	ms_get_safe_net_t ms;
	status = sgx_ecall(eid, 4, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

