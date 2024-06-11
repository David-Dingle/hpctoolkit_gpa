#ifndef TORCH_MONITOR_ADAPTOR_H
#define TORCH_MONITOR_ADAPTOR_H

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

#include <torch_monitor.h>


typedef enum adaptor_result {
  TORCH_MONITOR_ADAPTOR_SUCCESS = 0,
  TORCH_MONITOR_ADAPTOR_ERROR = 1
} adaptor_result_t;

/**
 * @brief
 **/
typedef uint64_t (*adaptor_get_id) ();

/**
 * @brief Register get_id function
 **/
EXTERNC adaptor_result_t adapter_get_id_register(adaptor_get_id func);

/**
 * @brief enable pytorch view analysis (log Python states)
 */
EXTERNC adaptor_result_t adaptor_torch_monitor_enable();


/**
 * @brief Invoke torch_monitor torch_monitor_python_state_get()
 * to update Python state.
 * 
 * @param max_num_states The max number of Python States you can get
 * @param states A pointor of a torch_monitor_python_state array
 * @param num_states An integer indicates the real size of the Python States
 * 
*/
EXTERNC adaptor_result_t python_state_get(size_t max_num_states, torch_monitor_python_state_t *states, size_t *num_states);


/**
 * @brief Invoke this function at both domain == CUPTI_CB_DOMAIN_DRIVER_API or CUPTI_CB_DOMAIN_RUNTIME_API
 * to assemble cct_node_t.persistent_id with Python states fetched by torch-monitor and log them in the file
 * 
 * @param cct_node_persistent_id cct_node_t.persistent_id from cupti_api.c
 * 
 */
EXTERNC adaptor_result_t callpath_assemble(int32_t cct_node_persistent_id);

#endif  // TORCH_MONITOR_ADAPTOR_H
