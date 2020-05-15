#include "Enclave_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define ADD_ASSIGN_OVERFLOW(a, b) (	\
	((a) += (b)) < (b)	\
)


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

static sgx_status_t SGX_CDECL sgx_Add_SafeLayer(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_Add_SafeLayer_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_Add_SafeLayer_t* ms = SGX_CAST(ms_Add_SafeLayer_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	Add_SafeLayer(ms->ms_input, ms->ms_output);


	return status;
}

static sgx_status_t SGX_CDECL sgx_FrontPropSafe(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_FrontPropSafe_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_FrontPropSafe_t* ms = SGX_CAST(ms_FrontPropSafe_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	FrontPropSafe(ms->ms_input);


	return status;
}

static sgx_status_t SGX_CDECL sgx_BackPropSafe(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_BackPropSafe_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_BackPropSafe_t* ms = SGX_CAST(ms_BackPropSafe_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	BackPropSafe(ms->ms_weight, ms->ms_delta);


	return status;
}

static sgx_status_t SGX_CDECL sgx_Opt_safe(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_Opt_safe_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_Opt_safe_t* ms = SGX_CAST(ms_Opt_safe_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	Opt_safe(ms->ms_type, ms->ms_learn_rate, ms->ms_decay_rate);


	return status;
}

static sgx_status_t SGX_CDECL sgx_get_safe_net(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_get_safe_net_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_get_safe_net_t* ms = SGX_CAST(ms_get_safe_net_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	ms->ms_retval = get_safe_net();


	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv;} ecall_table[5];
} g_ecall_table = {
	5,
	{
		{(void*)(uintptr_t)sgx_Add_SafeLayer, 0},
		{(void*)(uintptr_t)sgx_FrontPropSafe, 0},
		{(void*)(uintptr_t)sgx_BackPropSafe, 0},
		{(void*)(uintptr_t)sgx_Opt_safe, 0},
		{(void*)(uintptr_t)sgx_get_safe_net, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
} g_dyn_entry_table = {
	0,
};


