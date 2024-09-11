#include <linux/limits.h>  // PATH_MAX
#include <limits.h>    // PATH_MAX
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <hpcrun/cct/cct.h>

#include "torch_monitor_adaptor.h"

#ifdef DEBUG
#define PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define PRINT(...) 
#endif

#define MAX_NUM_STATES 30
static __thread size_t num_states;
static __thread torch_monitor_python_state_t python_states[MAX_NUM_STATES];
static FILE * fp;
static pthread_mutex_t mutex;

static adaptor_get_id update_id_func = NULL;
static char out_dir[PATH_MAX];


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
//start log Python States into gpa.log
      for(int i = 0; i < num_states; i++){
        printf("%s\n",python_states[i].file_name);
        printf("%s\n", python_states[i].function_name);
        printf("%ld : ",python_states[i].function_first_lineno);
        printf("%ld\n",python_states[i].lineno);
      }
      printf("----------------\n");
// end
    }
  }
}

/**
 * @brief enable pytorch view analysis (log Python states)
 */
adaptor_result_t adaptor_torch_monitor_enable(){

  torch_monitor_status_t status = torch_monitor_domain_enable(TORCH_MONITOR_DOMAIN_FUNCTION);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    fprintf(stderr, "Torch monitor status: %d\n", status);
    exit(1);
  }

  status = torch_monitor_domain_enable(TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    fprintf(stderr, "Torch monitor status: %d\n", status);
    exit(1);
  }

  status = torch_monitor_callback_subscribe(adaptor_callback);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    fprintf(stderr, "Torch monitor status: %d\n", status);
    exit(1);
  }
  status = torch_monitor_init();
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    fprintf(stderr, "Torch monitor status: %d\n", status);
    exit(1);
  }

  return TORCH_MONITOR_ADAPTOR_SUCCESS;
}

adaptor_result_t adaptor_stream_open(void){
  pthread_mutex_lock(&mutex);
  fp = fopen(out_dir, "a");
  pthread_mutex_unlock(&mutex);

  return TORCH_MONITOR_ADAPTOR_SUCCESS;
}

adaptor_result_t adaptor_stream_close(void){
  pthread_mutex_lock(&mutex);
  fclose(fp);
  pthread_mutex_unlock(&mutex);

  return TORCH_MONITOR_ADAPTOR_SUCCESS;
}

// the real assembler function
adaptor_result_t callpath_assemble_real(int32_t cct_node_persistent_id, uintptr_t lm_ip){
  if (num_states == 0){
    return TORCH_MONITOR_ADAPTOR_SUCCESS;
  }
  
  pthread_mutex_lock(&mutex);
  if (fp != NULL){
    uint64_t callpath_id = update_id_func();
    fprintf(fp, "id\n");
    fprintf(fp, "%lu\n",callpath_id);

    fprintf(fp, "ctx_id\n");
    fprintf(fp, "%d\n",cct_node_persistent_id);

    fprintf(fp, "lm_ip\n");
    fprintf(fp, "%ld\n",lm_ip);

    fprintf(fp, "num_states\n");
    fprintf(fp, "%lu\n",num_states);

    // malloc space for string concat
    char* all_states;
    char num_to_str[20];
    if (num_states != 0){
      all_states = (char*)malloc(4 * num_states * sizeof(char) * (strlen(python_states[0].file_name) + strlen(python_states[0].function_name)));
      *all_states = '\0';
    } // end

    for(int i = 0; i < num_states; i++){
      fprintf(fp, "file_name\n");
      fprintf(fp, "%s\n",python_states[i].file_name);

      fprintf(fp, "function_name\n");
      fprintf(fp, "%s\n", python_states[i].function_name);

      fprintf(fp, "function_first_lineno\n");
      fprintf(fp, "%ld\n",python_states[i].function_first_lineno);

      fprintf(fp, "lineno\n");
      fprintf(fp, "%ld\n",python_states[i].lineno);

      // start to print hashed pystates
      strcat(all_states, python_states[i].file_name);
      strcat(all_states, python_states[i].function_name);

      sprintf(num_to_str, "%ld", python_states[i].function_first_lineno);
      strcat(all_states, num_to_str);
      sprintf(num_to_str, "%ld", python_states[i].lineno);
      strcat(all_states, num_to_str);
      // end
    }
    fprintf(fp, "pystates_hash\n");
    fprintf(fp, "%s\n",all_states);

    free(all_states);
    // fprintf(fp, "----------------\n\n");

    pthread_mutex_unlock(&mutex);
  } else {  //fp ==NULL
    fprintf(stderr, "File stream status abnormal.\n");
    pthread_mutex_unlock(&mutex);
    return TORCH_MONITOR_ADAPTOR_ERROR;
  }

  return TORCH_MONITOR_ADAPTOR_SUCCESS;
}

// the public interface
adaptor_result_t callpath_assemble(gpu_activity_t * activity) {
  // read instruction identifier
  cct_node_t *cct_node = activity->cct_node;
  int32_t cct_node_persistent_id = hpcrun_cct_persistent_id(cct_node);
  gpu_pc_sampling_t *sinfo = &(activity->details.pc_sampling);
  // uint16_t lm_id = sinfo->pc.lm_id;
  uintptr_t lm_ip = sinfo->pc.lm_ip;
  // void * ip = hpcrun_denormalize_ip(&(sinfo->pc));

  // read 
  return callpath_assemble_real(cct_node_persistent_id, lm_ip); // stores the pystates, cct_node_t identifier, and the sampled instruction IP offline, together
}

adaptor_result_t adaptor_output_dir_config(const char *dir) {
  if (dir) {
    for (size_t i = 0; i < PATH_MAX; i++){
      out_dir[i] = *(dir + i);
    }
    sprintf(&out_dir[strlen(out_dir)], "%s", "torch_view_report.csv");
  }
  // printf("Loc: %s\n", out_dir);
  
  return TORCH_MONITOR_ADAPTOR_SUCCESS;
}
