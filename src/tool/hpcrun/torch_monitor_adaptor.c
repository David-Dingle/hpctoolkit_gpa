#include "torch_monitor_adaptor.h"

#define MAX_NUM_STATES 30
static __thread size_t num_states;
static __thread torch_monitor_python_state_t python_states[MAX_NUM_STATES];

#ifdef DEBUG
#define PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define PRINT(...) 
#endif

static adaptor_get_id update_id_func = NULL;


adaptor_result_t adapter_get_id_register(adaptor_get_id func) {
  update_id_func = func;
  return TORCH_MONITOR_ADAPTOR_SUCCESS;
}

adaptor_result_t python_state_get(size_t max_num_states, torch_monitor_python_state_t *states, size_t *num_states){
  torch_monitor_status_t status = torch_monitor_python_state_get(max_num_states, python_states, num_states);
  if(states == TORCH_MONITOR_STATUS_SUCCESS){
    return TORCH_MONITOR_ADAPTOR_SUCCESS;
  } else{
    PRINT("ERROR: torch-monitor returns statt: ", states);
    return TORCH_MONITOR_ADAPTOR_ERROR;
  }
}

static void adaptor_callback(torch_monitor_callback_site_t callback_site,
                            torch_monitor_callback_data_t* callback_data){
                              
  if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
    if (callback_data->domain != TORCH_MONITOR_DOMAIN_MEMORY) {
      python_state_get(MAX_NUM_STATES, python_states, &num_states);
    }
  }
}

/**
 * @brief enable pytorch view analysis (log Python states)
 */
adaptor_result_t adaptor_torch_monitor_enable(){

  torch_monitor_status_t status = torch_monitor_domain_enable(TORCH_MONITOR_DOMAIN_FUNCTION);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    // std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }

  status = torch_monitor_domain_enable(TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    // std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }

  status = torch_monitor_callback_subscribe(adaptor_callback);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    // std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }
  status = torch_monitor_init();
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    // std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }

  return TORCH_MONITOR_ADAPTOR_SUCCESS;
}

adaptor_result_t callpath_assemble(int32_t cct_node_persistent_id){
  printf("%d\n",cct_node_persistent_id);
  for(int i = 0; i < num_states; i++){
    printf("%s\n",python_states[i].file_name);
    printf("%s\n", python_states[i].function_name);
    printf("%ld : ",python_states[i].function_first_lineno);
    printf("%ld\n",python_states[i].lineno);
  }
  printf("----------------\n");

  return TORCH_MONITOR_ADAPTOR_SUCCESS;
}