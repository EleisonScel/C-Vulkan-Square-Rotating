/* Copyright 2026 EleisonScel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/assert_m.h"
#include "common/exit_print.h"
#include "common/safe_round.h"
#include "common/safe_alloc.h"
#include "common/ring_buffer.h"
#include "common/handle_file.h"
#include "common/clamp_values.h"
#include "common/vulkan_wrapped.h"
#include "common/aligned_memory.h"
#include "common/cleanup_register.h"
#include "common/safe_multiplication.h"
#include "common/write_out_error_message.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>	/* glfwInit	*/

#include <cglm/cglm.h>	/* glm_rad	*/

#include <pthread.h>	/* pthread_	*/
#include <limits.h>		/* INT_MAX	*/
#include <math.h>		/* fminf	*/

#include <stdio.h>		/* fprintf				*/
#include <stddef.h>		/* offsetof				*/
#include <string.h>		/* strcmp				*/
#include <stdlib.h>		/* EXIT_STATUS			*/
#include <stdint.h>		/* uint32_t				*/
#include <stdbool.h>	/* bool					*/
#include <stdatomic.h>	/* atomic_uint_least64_t*/

#ifndef NDEBUG
#	define VSR_DEBUG_LOG(format)		fprintf( stderr, (format"\n") )
#	define VSR_DEBUG_LOGF(format, ...)	fprintf( stderr, (format"\n"), __VA_ARGS__ )
#else
#	define VSR_DEBUG_LOG(format)		((void) 0)
#	define VSR_DEBUG_LOGF(format, ...) \
	do { if(0) fprintf( stderr, (format"\n"), __VA_ARGS__ ); } while(0)
#endif

#define VSR_WINDOW_WIDTH							800
#define VSR_WINDOW_HEIGHT							600

#define VSR_LIMIT_TURNOVER							6.2831853071f
#define VSR_SPIN_ANGLE_ROTATION						(VSR_LIMIT_TURNOVER / 256.f)

#define VSR_RESIZE_SETTLE_SECONDS					0.1
#define VSR_RESIZE_INVALID_FACTOR					4
#define VSR_LIMIT_EXTENT_MAXIMAL					(UINT32_MAX / VSR_RESIZE_INVALID_FACTOR)

#define VSR_MINIMUM_OF_MAXIMAL_IMAGE_DIMENSION_2D	4096

static_assert_m(
	VSR_LIMIT_EXTENT_MAXIMAL >= VSR_MINIMUM_OF_MAXIMAL_IMAGE_DIMENSION_2D,
	"extent limit shall not go below Vulkan specification minimum"
);

#define VSR_ATTACHMENT_COLOR_AMOUNT					1
#define VSR_EXTENSION_GROUPS_AMOUNT_INSTANCE		2
#define VSR_EXTENSION_GROUPS_AMOUNT_DEVICE			2
#define VSR_LIMIT_FRAMES_IN_FLIGHT					2
#define VSR_QUEUE_FAMILIES_AMOUNT					3
#define VSR_LIMIT_IMAGE_ACQUIRE_ATTEMPTS			16
#define VSR_LIMIT_FAILURES_SWAPCHAIN_RECREATE		16
#define VSR_LIMIT_FAILURES_FRAME_DISCARD			60
#define VSR_LIMIT_STACK_FAMILIES					64
#define VSR_LIMIT_STACK_EXTENSIONS					128
#define VSR_LIMIT_STACK_DELETION_QUEUE				256
#define VSR_LIMIT_TIME_WAIT_ACQUIRE					1000000000ULL /* 1	second	*/
#define VSR_LIMIT_TIME_WAIT_FENCE					10000000000ULL/* 10	seconds	*/

#define VSR_SPIRV_MAGIC_RECOGNITION_NUMBER			0x07230203

static_assert_m(
	VSR_LIMIT_FRAMES_IN_FLIGHT <= UINT32_MAX / 2,
	"frames in flight limit must not exceed half of uint32_t for a wrap-aware deletion"
);
static_assert_m(
	VK_MAX_MEMORY_TYPES <= 32, "Current max memory types allow undefined behavior"
);
static_assert_m(
	INT_MAX <= UINT32_MAX, "Positive integer must fit into uint32_t for frame buffer extent cast"
);

struct VSR_Extension_Names {
	const char * const 	* data_pointer;
	uint32_t			amount;
};

struct VSR_Extension_Names_Mutable {
	const char	** data_pointer;
	uint32_t	amount;
};

struct VSR_Extension_Properties {
	const struct VkExtensionProperties	* data_pointer;
	uint32_t							amount;
};

struct VSR_Extension_Properties_Mutable {
	struct VkExtensionProperties	* data_pointer;
	uint32_t						amount;
};

struct VSR_Buffer_Allocation_Data {
	VkBuffer		buffer;
	VkDeviceMemory	memory;
	bool			is_coherent;
};

struct VSR_Memory_Properties {
	VkMemoryPropertyFlags	list_required;
	VkMemoryPropertyFlags	list_forbidden;
};

struct VSR_Memory_Levels_Requirements {
	const struct VSR_Memory_Properties	* memory_properties;
	uint32_t							properties_amount;
};

struct VSR_Queue_Family_Indices {
	uint32_t	graphics_family;
	uint32_t	present_family;
	uint32_t	transfer_family;
	bool		has_graphics_family;
	bool		has_present_family;
	bool		has_transfer_family;
};

struct VSR_Deletion_Entity {
	VkImageView		* swap_chain_image_views_pointer;
	VkFramebuffer	* swap_chain_frame_buffer_pointer;
	VkFramebuffer	swap_chain_frame_buffer;
	VkSwapchainKHR	swap_chain;
	/* NULL on maintenance_1 or image_views_amount */
	VkSemaphore		* render_finished_semaphores_pointer;
	uint32_t		image_views_amount;
	uint32_t		delete_frame;
};

struct VSR_Swap_Chain_Support_Details {
	struct VkSurfaceFormatKHR		* surface_formats_pointer;
	VkPresentModeKHR				* present_modes_pointer;
	uint32_t						formats_amount;
	uint32_t						present_modes_amount;
	struct VkSurfaceCapabilitiesKHR	surface_capabilities;
};

struct VSR_Synchronization_Frame {
	VkSemaphore	image_available_semaphore;
	VkFence		in_flight_fence;
};

/* always must be changed with swap_chain_image_views_pointer */
struct VSR_Swap_Chain_Data {
	VkSwapchainKHR		swap_chain;
	/* imageless frame buffer */
	VkFramebuffer		frame_buffer;
	uint32_t			image_views_amount;
	struct VkExtent2D	extent;
	VkFormat			image_format;
	/* image_views_amount on maintenance_1 or NULL */
	VkFramebuffer		* frame_buffers_pointer;
};

struct VSR_Frame_State {
	bool	is_projection_dirty;
	bool	is_resize_pending;
	int		width;
	int		height;
};

struct VSR_Capabilities_Vulkan {
	PFN_vkGetPhysicalDeviceFeatures2KHR				get_physical_device_features_2;
	PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR	get_physical_device_surface_capabilities_2;
	bool											has_get_physical_device_properties_2;
	bool											has_surface_maintenance_1;
};

struct VSR_Capabilities_Device {
	bool has_swapchain_maintenance_1;
	bool has_present_scaling_stretch;
	bool has_imageless_frame_buffer;
};

struct VSR_Application {
	/* Hot data */
	VkDevice							device;
	VkQueue								graphics_queue;	/* implicitly destroyed with VkDevice */
	VkQueue								present_queue;	/* implicitly destroyed with VkDevice */
	struct VSR_Swap_Chain_Data			swap_chain_data;
	VkRenderPass						render_pass;
	VkPipelineLayout					pipeline_layout;
	VkPipeline							graphics_pipeline;
	VkBuffer							buffer_vertex;
	VkBuffer							buffer_index;

	/* backed on frames_in_flight_limit */
	VkBuffer							* buffers_uniform_pointer;
	VkDeviceMemory						buffers_uniform_memory;
	void								* buffers_uniform_mapped_pointer;

	/* handles are auto freed with its command pool destroying */
	VkDescriptorSet						* descriptor_sets_pointer;
	VkCommandBuffer						* command_buffers_pointer;

	/* frames_in_flight_limit */
	struct VSR_Synchronization_Frame	* synchronization_frame_pointer;
	/* render_finished_semaphores_amount */
	VkSemaphore							* render_finished_semaphores_pointer;
	uint32_t							render_finished_semaphores_amount;
	/* frames_in_flight_limit on maintenance_1 or NULL */
	VkFence								* present_fences_pointer;

	VkDeviceSize						buffer_uniform_size_alignment;
	VkDeviceSize						buffer_uniform_size_flush;
	VkDeviceSize						buffer_uniform_size;
	mat4								cached_view, cached_projection, cached_projection_view;

	float								spin_angle_current, spin_angle_rotation;

	/* for lower latency, otherwise back on VkImage.size */
	uint8_t								frames_in_flight_limit;
	uint8_t								current_frame;
	bool								is_buffer_uniform_coherent;
	atomic_bool							is_swap_chain_valid;
	atomic_bool							is_render_failed;
	bool								is_minimized;
	bool								is_running;
	struct VSR_Frame_State				frame_state;

	/* Warm data */
	VkImageView							* swap_chain_image_views_pointer;/*image count backed*/
	uint32_t							frame_counter;
	uint32_t							swap_chain_recreate_failed_amount;
	uint32_t							frame_discarded_amount;
	struct RB_Ring_Buffer				deletion_queue;
	struct VSR_Deletion_Entity			deletion_entities[VSR_LIMIT_STACK_DELETION_QUEUE];
	pthread_t							render_thread;
	pthread_mutex_t						render_mutex;
	pthread_cond_t						render_condition;

	/* Cold data */
	atomic_uint_least64_t				swap_chain_extent_packed;
	double								last_resize_time_seconds;
	GLFWwindow							* window_pointer;
	VkInstance							instance;
	VkSurfaceKHR						surface;
	VkPhysicalDevice					device_physical;
	VkQueue								transfer_queue;	/* implicitly destroyed with VkDevice */
	VkCommandPool						command_pool_graphic;
	VkCommandPool						command_pool_transfer;
	VkDeviceMemory						buffer_memory_vertex;
	VkDeviceMemory						buffer_memory_index;
	VkPhysicalDeviceMemoryProperties	memory_properties;
	uint32_t							image_dimension_2d_maximal;
	VkDescriptorPool					descriptor_pool;
	VkDescriptorSetLayout				descriptor_set_layout;
	struct VSR_Queue_Family_Indices		queue_family_indices;
	bool								is_initialized_glfw;
	bool								is_render_thread_created;
	bool								is_thread_objects_created;
	struct VSR_Capabilities_Vulkan		capabilities_vulkan;
	struct VSR_Capabilities_Device		capabilities_device;
#ifndef NDEBUG
	bool								is_debug_messenger_established;
	VkDebugUtilsMessengerEXT			debug_messenger_function;
#endif
};

struct VSR_Device_Candidate {
	uint32_t						score;
	uint32_t						image_dimension_2d_maximal;
	VkPhysicalDevice				device;
	struct VSR_Capabilities_Device	capabilities;
	struct VSR_Queue_Family_Indices	queue_family_indices;
};

struct VSR_Extension_Group {
	struct VSR_Extension_Names	names;
	bool						* is_available_pointer;
};

struct VSR_Extension_Groups {
	struct VSR_Extension_Group	* data_pointer;
	uint32_t					amount;
};

/* extensions: instance */
static const char * const global_instance_extensions_device_properties_data[] = {
	VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
};
static const struct VSR_Extension_Names global_instance_extensions_device_properties = {
	.data_pointer	= global_instance_extensions_device_properties_data,
	.amount			=
		sizeof(global_instance_extensions_device_properties_data) /
		sizeof(global_instance_extensions_device_properties_data[0])
};

static const char * const global_instance_extensions_surface_capabilities_data[] = {
	VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
	VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME
};
static const struct VSR_Extension_Names global_instance_extensions_surface_capabilities = {
	.data_pointer	= global_instance_extensions_surface_capabilities_data,
	.amount			=
		sizeof(global_instance_extensions_surface_capabilities_data) /
		sizeof(global_instance_extensions_surface_capabilities_data[0])
};

/* extensions: device */
static const char * const global_device_extensions_swapchain_maintenance_1_data[] = {
	VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME
};
#define VSR_EXTENSIONS_AMOUNT_DEVICE_SWAPCHAIN_MAINTENANCE_1		\
	sizeof(global_device_extensions_swapchain_maintenance_1_data) /	\
	sizeof(global_device_extensions_swapchain_maintenance_1_data[0])
static const struct VSR_Extension_Names global_device_extensions_swapchain_maintenance_1 = {
	.data_pointer	= global_device_extensions_swapchain_maintenance_1_data,
	.amount			= VSR_EXTENSIONS_AMOUNT_DEVICE_SWAPCHAIN_MAINTENANCE_1
};

static const char * const global_device_extensions_imageless_frame_buffer_data[] = {
	VK_KHR_IMAGELESS_FRAMEBUFFER_EXTENSION_NAME,
	VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,
	VK_KHR_MAINTENANCE2_EXTENSION_NAME
};
#define VSR_EXTENSIONS_AMOUNT_DEVICE_IMAGELESS_FRAME_BUFFER			\
	sizeof(global_device_extensions_imageless_frame_buffer_data) /	\
	sizeof(global_device_extensions_imageless_frame_buffer_data[0])
static const struct VSR_Extension_Names global_device_extensions_imageless_frame_buffer = {
	.data_pointer	= global_device_extensions_imageless_frame_buffer_data,
	.amount			= VSR_EXTENSIONS_AMOUNT_DEVICE_IMAGELESS_FRAME_BUFFER
};

static const char * const global_device_extensions_required_data[] = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
#define VSR_EXTENSIONS_AMOUNT_DEVICE_REQUIRED		\
	sizeof(global_device_extensions_required_data) /\
	sizeof(global_device_extensions_required_data[0])
static const struct VSR_Extension_Names global_device_extensions_required = {
	.data_pointer	= global_device_extensions_required_data,
	.amount			= VSR_EXTENSIONS_AMOUNT_DEVICE_REQUIRED
};
#define VSR_EXTENSIONS_AMOUNT_DEVICE_MAXIMAL				\
	VSR_EXTENSIONS_AMOUNT_DEVICE_SWAPCHAIN_MAINTENANCE_1+	\
	VSR_EXTENSIONS_AMOUNT_DEVICE_IMAGELESS_FRAME_BUFFER + VSR_EXTENSIONS_AMOUNT_DEVICE_REQUIRED

struct VSR_Vertex {
	vec2 position;
	vec3 color;
};

struct VSR_Uniform_Buffer_Object {
	mat4 model;
	mat4 view_projection;
};

static const uint16_t global_indices_array[] = {
	0, 1, 2, 2, 3, 0
};

#ifndef NDEBUG
static bool global_is_validation_layer_supported = false;

static const char * global_validation_layers_data[] = {
	"VK_LAYER_KHRONOS_validation"
};
static const struct VSR_Extension_Names global_validation_layers = {
	.data_pointer	= global_validation_layers_data,
	.amount			=
		sizeof(global_validation_layers_data) / sizeof(global_validation_layers_data[0])
};
#endif /* NDEBUG */

/* common */
static bool vsr_application_initialize(struct VSR_Application * restrict application_pointer);

/* window specific */
static void vsr_callback_glfw_key(GLFWwindow * window_pointer, int key, int scancode, int action, int mods);
static void vsr_callback_glfw_error(int error_code, const char * description_pointer);
static void vsr_callback_glfw_window_iconify(GLFWwindow * window_pointer, int iconified);
static void vsr_callback_glfw_frame_buffer_size(GLFWwindow * window_pointer, int width, int height);
static bool vsr_window_initialize(struct VSR_Application * restrict application_pointer);

/* Vulkan Initialization */
static bool vsr_surface_create(struct VSR_Application * restrict application_pointer);
static bool vsr_instance_create(struct VSR_Application * restrict application_pointer);
static bool vsr_vulkan_initialize(struct VSR_Application * restrict application_pointer);

/* Vulkan Debug messenger */
#ifndef NDEBUG
static void vsr_debug_utils_messenger_extension_destroy(VkInstance instance, VkDebugUtilsMessengerEXT debug_messenger);
static void vsr_debug_messenger_create_information_populate(struct VkDebugUtilsMessengerCreateInfoEXT * restrict creation_information_pointer);
static bool vsr_debug_messenger_setup(struct VSR_Application * restrict application_pointer);
static bool vsr_validation_layer_support_check(void);
static VkResult vsr_debug_utils_messenger_extension_create(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT * restrict create_information_pointer, VkDebugUtilsMessengerEXT * restrict debug_messenger_pointer);
static VKAPI_ATTR VkBool32 VKAPI_CALL vsr_debug_callback_function(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, VkDebugUtilsMessageTypeFlagsEXT message_type, const VkDebugUtilsMessengerCallbackDataEXT * restrict data_callback_pointer, void * restrict data_user_pointer);
static void vsr_debug_gpu_print(VkPhysicalDeviceProperties device_properties, struct VSR_Capabilities_Device capabilities_device, uint32_t scores);
static inline const char * vsr_debug_device_type_print(const VkPhysicalDeviceType device_type);
static inline const char * vsr_debug_maintainability_print(bool flag);
#endif

/* extensions lists */
static void vsr_device_extensions_groups_fill(struct VSR_Capabilities_Device * restrict capabilities_device_pointer, struct VSR_Extension_Group out_groups_extension_array[static VSR_EXTENSION_GROUPS_AMOUNT_DEVICE]);
static void vsr_extensions_groups_availability(const struct VSR_Extension_Groups groups, const struct VSR_Extension_Properties extensions_available);
static void vsr_instance_extensions_groups_fill(struct VSR_Application * restrict application_pointer, struct VSR_Extension_Group out_groups_extension_array[static VSR_EXTENSION_GROUPS_AMOUNT_INSTANCE]);
static void vsr_device_extensions_enabled_array_fill(struct VSR_Application * restrict application_pointer, struct VSR_Extension_Names_Mutable * restrict out_extensions_pointer);
static const char * vsr_get_extensions_available(struct VSR_Extension_Properties_Mutable * restrict out_extensions_pointer);
static const char * vsr_instance_extensions_check(struct VSR_Application * restrict application_pointer, struct VSR_Extension_Names_Mutable * restrict out_extensions_pointer);
static const char * vsr_instance_extensions_all_build(const struct VSR_Extension_Names extensions_required, const struct VSR_Extension_Group out_groups_extension_array[static VSR_EXTENSION_GROUPS_AMOUNT_INSTANCE], struct VSR_Extension_Names_Mutable * restrict out_extensions_pointer);
static const char * vsr_get_instance_extensions_required(struct VSR_Extension_Names * restrict extensions_required_pointer);
static inline bool vsr_extensions_available_check(const struct VSR_Extension_Names extensions_required, const struct VSR_Extension_Properties extensions_available);
static inline bool vsr_instance_extensions_required_check(const struct VSR_Extension_Names extensions_required, const struct VSR_Extension_Properties extensions_available);
static inline struct VSR_Extension_Properties vsr_extension_properties_freeze(const struct VSR_Extension_Properties_Mutable array_extension_properties_mutable);
/* pick up GPUs */
static void vsr_device_extension_required_check(VkPhysicalDevice device, PFN_vkGetPhysicalDeviceFeatures2KHR function_get_features_2, struct VSR_Capabilities_Device * restrict out_capabilities_pointer);
static bool vsr_device_extensions_get(VkPhysicalDevice device, struct VkExtensionProperties out_extensions_array_stack[static VSR_LIMIT_STACK_EXTENSIONS], struct VSR_Extension_Properties_Mutable * restrict out_extensions_pointer);
static bool vsr_device_physical_select(struct VSR_Application * restrict application_pointer);
static bool vsr_device_capabilities_build(VkPhysicalDevice device, VkSurfaceKHR surface, const struct VSR_Capabilities_Vulkan * restrict instance_capabilities_pointer, struct VSR_Capabilities_Device * restrict out_capabilities_device_pointer);
static bool vsr_device_present_scaling_stretch_check(VkPhysicalDevice device, VkSurfaceKHR surface, const struct VSR_Capabilities_Vulkan * restrict instance_capabilities_pointer);
static const char * vsr_queue_families_find(VkSurfaceKHR surface, VkPhysicalDevice device, struct VSR_Queue_Family_Indices * restrict out_queue_family_indices_pointer);
static const char * vsr_device_suitability_rate(struct VSR_Application * restrict application_pointer, VkPhysicalDevice device, struct VSR_Device_Candidate * restrict out_candidate_pointer);
static const char * vsr_surface_is_support_available(VkSurfaceKHR surface, VkPhysicalDevice device, uint32_t * restrict formats_amount_pointer, uint32_t * restrict present_modes_amount_pointer);
static inline bool vsr_queue_family_indices_is_complete(struct VSR_Queue_Family_Indices * restrict queue_family_indices_pointer);
/* device */
static const char * vsr_device_create(struct VSR_Application * restrict application_pointer, void * restrict features_chain_pointer, uint32_t families_amount, struct VkDeviceQueueCreateInfo * restrict queue_create_informations_array);
static const char * vsr_device_logical_create(struct VSR_Application * restrict application_pointer);
static const char * vsr_device_resources_create(struct VSR_Application * restrict application_pointer);
static bool vsr_device_recreate(struct VSR_Application * restrict application_pointer);
/* choose a swap chain part */
static void vsr_swap_chain_support_details_free(struct VSR_Swap_Chain_Support_Details * restrict swap_chain_support_pointer);
static inline void vsr_swap_chain_extent_write(struct VSR_Application * restrict application_pointer, struct VkExtent2D * restrict out_extent_pointer, struct VkSurfaceCapabilitiesKHR * restrict out_surface_capabilities_pointer);
static const char * vsr_swap_chain_create(struct VSR_Application * restrict application_pointer, struct VSR_Swap_Chain_Data * restrict out_swap_chain_data_pointer, VkImage ** restrict out_swap_chain_images_pointer);
static const char * vsr_swap_chain_support_query(VkSurfaceKHR surface, VkPhysicalDevice device, struct VSR_Swap_Chain_Support_Details * restrict out_swap_chain_support_details_pointer);
static struct VkExtent2D vsr_swap_extent_choose(const struct VkSurfaceCapabilitiesKHR * restrict surface_capabilities_pointer, uint32_t frame_buffer_width, uint32_t frame_buffer_height);
static struct VkSurfaceFormatKHR vsr_swap_surface_format_choose(const struct VkSurfaceFormatKHR * restrict available_formats_pointer, size_t available_formats_amount);
static inline const char * vsr_swap_chain_support_check(const struct VSR_Swap_Chain_Support_Details * restrict swap_chain_support_pointer, const uint32_t image_dimension_2d_maximal);
/* swap chain images */
static void vsr_image_views_destroy(VkDevice device, VkImageView * restrict image_views_pointer, uint32_t image_views_amount);
static const char * vsr_image_views_create(struct VSR_Application * restrict application_pointer, VkImage * restrict swap_chain_images_pointer, uint32_t swap_chain_image_views_amount, VkFormat swap_chain_image_format, VkImageView ** out_swap_chain_image_views_pointer);
/* swap chain recreation */
static void vsr_swap_chain_recreate_data_commit(struct VSR_Application * restrict application_pointer, const struct VSR_Swap_Chain_Data * restrict swap_chain_data_pointer, VkImageView * restrict swap_chain_image_views_pointer);
static bool vsr_swap_chain_recreate(struct VSR_Application * restrict application_pointer);
static bool vsr_swap_chain_recreate_data(struct VSR_Application * restrict application_pointer);
static bool vsr_swap_chain_is_extent_needs_update(struct VSR_Application * restrict application_pointer);
static const char * vsr_swap_chain_deletion_resources_handle(struct VSR_Application * restrict application_pointer, const uint32_t image_views_amount);
static const char * vsr_swap_chain_render_finished_semaphores_recreate(struct VSR_Application * restrict application_pointer, const uint32_t image_views_amount);
/* render */
static const char * vsr_render_pass_create(struct VSR_Application * restrict application_pointer);
/* pipeline */
static const char * vsr_graphics_pipeline_create(struct VSR_Application * restrict application_pointer);
static const char * vsr_graphics_pipeline_from_shaders_create(struct VSR_Application * restrict application_pointer, const VkShaderModule shader_module_vertex, const VkShaderModule shader_module_fragment);
/* describe descriptor of bindings in shaders */
static const char * vsr_descriptor_set_layout_create(struct VSR_Application * restrict application_pointer);
/* descriptor */
static const char * vsr_descriptor_pool_create(struct VSR_Application * restrict application_pointer);
/* descriptor set */
static const char * vsr_descriptor_sets_create(struct VSR_Application * restrict application_pointer);
/* shaders */
static const char * vsr_shader_module_create( VkDevice device, char * restrict shader_code_source_pointer, size_t file_size, VkShaderModule * restrict out_shader_module_pointer);
/* frame_buffers */
static const char * vsr_frame_buffer_create(struct VSR_Application * restrict application_pointer, VkImageView * restrict swap_chain_image_views_pointer, struct VSR_Swap_Chain_Data * restrict out_swap_chain_data_pointer);
static const char * vsr_frame_buffer_create_imaged(struct VSR_Application * restrict application_pointer, VkImageView * restrict swap_chain_image_views_pointer, struct VSR_Swap_Chain_Data * restrict swap_chain_data_pointer, struct VkFramebufferCreateInfo frame_buffer_create_information);
/* commands handle */
static bool vsr_command_buffer_record(struct VSR_Application * restrict application_pointer, VkCommandBuffer command_buffer, uint32_t image_index);
static const char * vsr_command_pools_create(struct VSR_Application * restrict application_pointer);
static const char * vsr_command_buffers_create(struct VSR_Application * restrict application_pointer);
static const char * vsr_inclusive_command_pool_create(struct VSR_Application * restrict application_pointer, VkCommandPool * restrict command_pool_pointer, VkCommandPoolCreateFlags flags, uint32_t family);
/* buffers */
static void vsr_buffer_copy_barrier(VkCommandBuffer command_buffer, VkBuffer buffer_destination, VkDeviceSize size, uint32_t graphics_family, uint32_t transfer_family);
static inline bool vsr_memory_type_is_coherent(const struct VSR_Application * restrict application_pointer, const uint32_t memory_type_index);
static const char * vsr_buffer_copy(struct VSR_Application * restrict application_pointer, VkBuffer buffer_source, VkBuffer buffer_destination, VkDeviceSize size, VkCommandPool command_pool);
static const char * vsr_buffer_create(struct VSR_Application * restrict application_pointer, VkDeviceSize size, VkBufferUsageFlags usage, const struct VSR_Memory_Levels_Requirements * restrict requirements_list, struct VSR_Buffer_Allocation_Data * restrict out_allocation_data_pointer);
static const char * vsr_buffers_fast_create(struct VSR_Application * restrict application_pointer);
static const char * vsr_buffer_uniform_create(struct VSR_Application * restrict application_pointer, const VkPhysicalDeviceProperties * restrict device_properties_pointer);
static const char * vsr_buffer_uniforms_create(struct VSR_Application * restrict application_pointer);
static const char * vsr_buffer_inclusive_create(struct VSR_Application * restrict application_pointer, const void * restrict buffer_pointer, VkDeviceSize buffer_size, VkBufferUsageFlags buffer_usage, struct VSR_Buffer_Allocation_Data allocation_data_staging, VkCommandPool command_pool, struct VSR_Buffer_Allocation_Data * restrict out_allocation_data_pointer);
static const char * vsr_buffer_ownership_acquire(struct VSR_Application * restrict application_pointer, VkBuffer buffer_destination, VkDeviceSize size);
static const char * vsr_buffer_ownership_acquire_record(struct VSR_Application * restrict application_pointer, VkCommandBuffer command_buffer, VkBuffer buffer_destination, VkDeviceSize size);
static const char * vsr_buffer_ownership_acquire_submit_wait(struct VSR_Application * restrict application_pointer, VkCommandBuffer command_buffer);
static inline const char * vsr_buffer_staging_create(struct VSR_Application * restrict application_pointer, VkDeviceSize buffer_size, struct VSR_Buffer_Allocation_Data * restrict out_allocation_data_pointer);
/* update uniform buffer data */
static void vsr_projection_refresh(struct VSR_Application * restrict application_pointer);
static bool vsr_buffer_uniform_update(struct VSR_Application * restrict application_pointer, uint32_t current_frame);
/* binding buffer description */
static struct VkVertexInputBindingDescription vsr_get_binding_description(void);
/* find memory type on GPU */
static bool vsr_memory_type_find(struct VSR_Application * restrict application_pointer, uint32_t type_filter, const struct VSR_Memory_Levels_Requirements * restrict memory_requirements_pointer, uint32_t * restrict out_type_index_pointer);
/* synchronization handle */
static bool vsr_synchronization_fence_present_recreate(struct VSR_Application * restrict application_pointer);
static void vsr_synchronization_fence_present_destroy(VkDevice device, VkFence * restrict fences_pointer, uint32_t fences_amount);
static void vsr_synchronization_semaphores_render_finished_destroy(VkDevice device, VkSemaphore * restrict semaphores_pointer, uint32_t semaphores_amount);
static const char * vsr_synchronization_create(struct VSR_Application * restrict application_pointer);
static const char * vsr_synchronization_frame_create(const struct VSR_Application * restrict application_pointer, struct VSR_Synchronization_Frame * restrict synchronization_frame_pointer);
static const char * vsr_synchronization_frames_create(struct VSR_Application * restrict application_pointer);
static const char * vsr_synchronization_fence_present_create(struct VSR_Application * restrict application_pointer, VkFence ** restrict out_fences_pointer, uint32_t fences_to_create_amount);
static const char * vsr_synchronization_semaphores_render_finished_create(struct VSR_Application * restrict application_pointer, VkSemaphore ** restrict out_semaphores_pointer, uint32_t semaphores_to_create_amount);
/* delayed deletion */
static void vsr_delay_deletion_process(struct VSR_Application * restrict application_pointer);
static void vsr_delay_deletion_data_destroy(struct VSR_Application * restrict application_pointer);
static const char * vsr_delay_deletion_cleanup(struct VSR_Application * restrict application_pointer);
static const char * vsr_delay_deletion_initialize(struct VSR_Application * restrict application_pointer);
static inline bool vsr_frame_reached(uint32_t frame_current, uint32_t frame_target);
static inline struct VSR_Deletion_Entity vsr_get_delay_deletion_entity_data(struct VSR_Application * restrict application_pointer);

/* draw frame */
static void vsr_frame_draw(struct VSR_Application * restrict application_pointer);
static void vsr_frame_discard(struct VSR_Application * restrict application_pointer, uint32_t image_index);
static void vsr_frame_render_failed(struct VSR_Application * restrict application_pointer, const char * restrict error_message_pointer);
static bool vsr_surface_recreate(struct VSR_Application * restrict application_pointer);
static bool vsr_frame_image_present(struct VSR_Application * restrict application_pointer, uint32_t image_index);
static bool vsr_frame_image_acquire(struct VSR_Application * restrict application_pointer, const struct VSR_Synchronization_Frame * restrict synchronization_frame_pointer, uint32_t * restrict out_image_index);
static bool vsr_frame_commands_submit(struct VSR_Application * restrict application_pointer, const struct VSR_Synchronization_Frame * restrict synchronization_frame_pointer, uint32_t image_index);
static bool vsr_frame_fence_present_reset(struct VSR_Application * restrict application_pointer);
static bool vsr_frame_image_present_result_handle(struct VSR_Application * restrict application_pointer, VkResult result);
static VkResult vsr_frame_fences_wait(const struct VSR_Application * restrict application_pointer, const struct VSR_Synchronization_Frame * restrict synchronization_frame_pointer);
static VkSemaphore vsr_frame_get_render_finished_semaphore(const struct VSR_Application * restrict application_pointer, uint32_t frame_index, uint32_t image_index);

/* app main cycle */
static void vsr_main_loop(struct VSR_Application * restrict application_pointer);
static void vsr_thread_render_stop(struct VSR_Application * restrict application_pointer);
static void * vsr_thread_render_function(void * restrict argument_pointer);

/* app clean up */
static void vsr_swap_chain_cleanup(struct VSR_Application * restrict application_pointer);
static void vsr_application_cleanup(void * restrict argument_pointer);
static void vsr_frame_buffer_destroy(VkDevice device, VkFramebuffer frame_buffer, VkFramebuffer * restrict frame_buffers_pointer, uint32_t frame_buffer_amount);
static void vsr_deletion_entity_destroy(struct VSR_Application * restrict application_pointer, struct VSR_Deletion_Entity * restrict entity_pointer);
static void vsr_device_and_resources_destroy(struct VSR_Application * restrict application_pointer);
static void vsr_synchronization_frames_destroy(struct VSR_Application * restrict application_pointer, uint8_t objects_amount);
static void vsr_swap_chain_data_associated_destroy(struct VSR_Application * restrict application_pointer);

int main (void) {
	struct VSR_Application application = { 0 };
	bool is_everything_went_well = vsr_application_initialize( &application );

	if(	is_everything_went_well == true ) {
		vsr_main_loop( &application );
		is_everything_went_well = atomic_load_explicit(
			&application.is_render_failed, memory_order_relaxed
		) == false;
	}

	exit( is_everything_went_well == true ? EXIT_SUCCESS : EXIT_FAILURE );
}

static void * vsr_thread_render_function( void * restrict argument_pointer ) {
	struct VSR_Application * application_pointer = (struct VSR_Application *) argument_pointer;
	assert_m( application_pointer != NULL, "No application found" );

	pthread_mutex_lock( &application_pointer->render_mutex );

	while ( application_pointer->is_running == true ) {
		if(	application_pointer->is_minimized == true ) {
			pthread_cond_wait(
				&application_pointer->render_condition, &application_pointer->render_mutex
			);
			continue;
		}

		bool is_resize_calm_down = false;
		if ( application_pointer->frame_state.is_resize_pending == true &&
			(glfwGetTime() - application_pointer->last_resize_time_seconds) >=
				VSR_RESIZE_SETTLE_SECONDS )
		{
			if(	vsr_swap_chain_is_extent_needs_update(application_pointer) == true )
				is_resize_calm_down = true;
			else
				application_pointer->frame_state.is_resize_pending = false;
		}

		if(	application_pointer->frame_state.is_projection_dirty == true )
			vsr_projection_refresh( application_pointer );

		pthread_mutex_unlock(&application_pointer->render_mutex );

		if(	is_resize_calm_down == true )
			vsr_swap_chain_recreate( application_pointer );
		vsr_frame_draw( application_pointer );

		pthread_mutex_lock( &application_pointer->render_mutex );
	}

	pthread_mutex_unlock( &application_pointer->render_mutex );

	return NULL;
}

static bool vsr_vulkan_initialize( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	if(	vsr_instance_create( application_pointer ) == false ) return false;

#ifndef NDEBUG

	if(	global_is_validation_layer_supported == true ) {
		if(	vsr_debug_messenger_setup( application_pointer ) == false )
			VSR_DEBUG_LOG("(vsr_vulkan_initialize) failed to set up debug messenger");
		else
			application_pointer->is_debug_messenger_established = true;
	}

#endif

	if(	vsr_surface_create(			application_pointer ) == false ) return false;
	if(	vsr_device_physical_select(	application_pointer ) == false ) return false;

	const char * error_message_pointer = NULL;
	if(	(error_message_pointer = vsr_device_logical_create(	application_pointer )) != NULL ||
		(error_message_pointer = vsr_device_resources_create(application_pointer)) != NULL)
	{
		woem_push( "%s", error_message_pointer );
		return false;
	}

	return true;
}

static const char * vsr_device_resources_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	const char * error_message_pointer = NULL;
	VkImage * swap_chain_images_pointer;

	if(	(error_message_pointer = vsr_delay_deletion_initialize(application_pointer)) != NULL ||
		(error_message_pointer = vsr_swap_chain_create(
			application_pointer, &application_pointer->swap_chain_data, &swap_chain_images_pointer
		)) != NULL )
		return error_message_pointer;

	if((error_message_pointer = vsr_image_views_create(
			application_pointer, swap_chain_images_pointer,
			application_pointer->swap_chain_data.image_views_amount,
			application_pointer->swap_chain_data.image_format,
			&application_pointer->swap_chain_image_views_pointer
		)) != NULL ||
		(error_message_pointer = vsr_render_pass_create(application_pointer))			!= NULL ||
		(error_message_pointer = vsr_descriptor_set_layout_create(application_pointer))	!= NULL ||
		(error_message_pointer = vsr_graphics_pipeline_create(application_pointer))		!= NULL ||
		(error_message_pointer = vsr_frame_buffer_create(
			application_pointer, application_pointer->swap_chain_image_views_pointer,
			&application_pointer->swap_chain_data
		)) != NULL ||
		(error_message_pointer = vsr_command_pools_create(application_pointer))		!= NULL ||
		(error_message_pointer = vsr_buffers_fast_create(application_pointer))		!= NULL ||
		(error_message_pointer = vsr_buffer_uniforms_create(application_pointer))	!= NULL ||
		(error_message_pointer = vsr_descriptor_pool_create(application_pointer))	!= NULL ||
		(error_message_pointer = vsr_descriptor_sets_create(application_pointer))	!= NULL ||
		(error_message_pointer = vsr_command_buffers_create(application_pointer))	!= NULL ||
		(error_message_pointer = vsr_synchronization_create(application_pointer))	!= NULL )
		goto out;

	atomic_store_explicit(
		&application_pointer->is_swap_chain_valid, true, memory_order_release
	);

out:
	free( swap_chain_images_pointer );
	return error_message_pointer;
}

static const char * vsr_synchronization_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	const char * error_message_pointer;
	if((error_message_pointer = vsr_synchronization_frames_create(application_pointer)) != NULL )
		return error_message_pointer;

	application_pointer->render_finished_semaphores_amount =
		(application_pointer->capabilities_device.has_swapchain_maintenance_1 == true)
		? application_pointer->frames_in_flight_limit
		: application_pointer->swap_chain_data.image_views_amount;

	if((error_message_pointer = vsr_synchronization_semaphores_render_finished_create(
			application_pointer, &application_pointer->render_finished_semaphores_pointer,
			application_pointer->render_finished_semaphores_amount
		)) != NULL )
	{
		application_pointer->render_finished_semaphores_pointer = NULL;
		return error_message_pointer;
	}

	if(	application_pointer->capabilities_device.has_swapchain_maintenance_1 == true )
		return vsr_synchronization_fence_present_create(
			application_pointer, &application_pointer->present_fences_pointer,
			application_pointer->frames_in_flight_limit
		);

	return NULL;
}

static void vsr_main_loop( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	while ( glfwWindowShouldClose( application_pointer->window_pointer ) == false &&
			atomic_load_explicit(
				&application_pointer->is_render_failed, memory_order_relaxed
			) == false )
		glfwWaitEvents();

	vsr_thread_render_stop( application_pointer );
}

static void vsr_thread_render_stop(struct VSR_Application * restrict application_pointer) {
	assert_m( application_pointer != NULL, "No application found" );
	assert_m(
		application_pointer->is_render_thread_created == true,
		"function shall be called after threads creation only"
	);

	pthread_mutex_lock(&application_pointer->render_mutex);

	application_pointer->is_running = false;

	pthread_cond_broadcast(&application_pointer->render_condition);
	pthread_mutex_unlock(&application_pointer->render_mutex);

	pthread_join( application_pointer->render_thread, NULL );
	application_pointer->is_render_thread_created = false;
}

static bool vsr_application_initialize(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	cr_register_cleanup_wrapper( vsr_application_cleanup, application_pointer );

	*application_pointer = (struct VSR_Application) {
		.frame_state			= (struct VSR_Frame_State) {
			.is_projection_dirty= true,
			.width				= VSR_WINDOW_WIDTH,
			.height				= VSR_WINDOW_HEIGHT,
		},
		.frames_in_flight_limit	= VSR_LIMIT_FRAMES_IN_FLIGHT,
		.spin_angle_rotation	= VSR_SPIN_ANGLE_ROTATION,
		.is_running				= true
	};

	atomic_init( &application_pointer->is_swap_chain_valid,			false	);
	atomic_init( &application_pointer->is_render_failed,			false	);
	atomic_init( &application_pointer->swap_chain_extent_packed,	0		);

	if(	pthread_mutex_init( &application_pointer->render_mutex, NULL ) != 0 ) {
		woem_push(
			"(vsr_application_initialize) render mutual exclusion initialization failed"
		);
		return false;
	}
	if(	pthread_cond_init( &application_pointer->render_condition, NULL ) != 0 ) {
		woem_push( "(vsr_application_initialize) render condition initialization failed" );
		pthread_mutex_destroy( &application_pointer->render_mutex );
		return false;
	}

	application_pointer->is_thread_objects_created = true;
	if(	vsr_window_initialize( application_pointer ) == false ||
		vsr_vulkan_initialize( application_pointer ) == false )
		return false;

	glm_lookat(
		(vec3){ 2.f, 2.f, 2.f },
		(vec3){ 0.f, 0.f, 0.f },
		(vec3){ 0.f, 0.f, 1.f },
		application_pointer->cached_view
	);

	if(	pthread_create(
			&application_pointer->render_thread, NULL, vsr_thread_render_function,
			application_pointer
		) != 0 )
	{
		woem_push( "(vsr_application_initialize) render thread creation failed" );
		return false;
	}

	application_pointer->is_render_thread_created = true;
	return true;
}

static bool vsr_device_recreate( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	vsr_device_and_resources_destroy( application_pointer );

	const char * error_message_pointer = NULL;
	if(	(error_message_pointer = vsr_device_logical_create( application_pointer )) != NULL ||
		(error_message_pointer = vsr_device_resources_create(application_pointer)) != NULL )
	{
		VSR_DEBUG_LOGF( "%s", error_message_pointer );
		return false;
	}

	application_pointer->frame_discarded_amount = 0;
	return true;
}

static void vsr_swap_chain_data_associated_destroy(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	vkDeviceWaitIdle(application_pointer->device);
	vsr_delay_deletion_data_destroy(application_pointer);
	vsr_swap_chain_cleanup(application_pointer);

	application_pointer->frame_counter = 0;
	atomic_store_explicit(
		&application_pointer->is_swap_chain_valid, false, memory_order_relaxed
	);
}

static void vsr_delay_deletion_data_destroy(struct VSR_Application * restrict application_pointer)
{
	assert_m( application_pointer != NULL, "No application found" );

	struct VSR_Deletion_Entity * entity_pointer;
	while( (entity_pointer = rb_ring_buffer_pop(&application_pointer->deletion_queue)) != NULL )
		vsr_deletion_entity_destroy( application_pointer, entity_pointer );
}

static const char * vsr_delay_deletion_initialize(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	return rb_ring_buffer_initialize_static(
				&application_pointer->deletion_queue, application_pointer->deletion_entities,
				VSR_LIMIT_STACK_DELETION_QUEUE, sizeof(struct VSR_Deletion_Entity), 0, 0, 0
			) == false
		? "(vsr_delay_deletion_initialize) ring buffer initialization failed"
		: NULL;
}

static void vsr_device_and_resources_destroy(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	if(	application_pointer->device == VK_NULL_HANDLE )
		return;

	vsr_swap_chain_data_associated_destroy(application_pointer);

	if(	application_pointer->command_buffers_pointer != NULL ) {
		free( application_pointer->command_buffers_pointer );
		application_pointer->command_buffers_pointer = NULL;
	}

	if(	application_pointer->buffers_uniform_pointer != NULL ) {
		for ( uint8_t buffer_uniform_index = 0;
				buffer_uniform_index < application_pointer->frames_in_flight_limit;
				++buffer_uniform_index )
			vkDestroyBuffer(
				application_pointer->device,
				application_pointer->buffers_uniform_pointer[buffer_uniform_index], NULL
			);
		free( application_pointer->buffers_uniform_pointer );
		application_pointer->buffers_uniform_pointer = NULL;
	}

	if(	application_pointer->buffers_uniform_memory != VK_NULL_HANDLE ) {
		if(	application_pointer->buffers_uniform_mapped_pointer != NULL ) {
			vkUnmapMemory(
				application_pointer->device, application_pointer->buffers_uniform_memory
			);
			application_pointer->buffers_uniform_mapped_pointer = NULL;
		}
		vkFreeMemory(
			application_pointer->device, application_pointer->buffers_uniform_memory, NULL
		);
		application_pointer->buffers_uniform_memory = NULL;
	}

	if(	application_pointer->descriptor_pool != VK_NULL_HANDLE ) {
		vkDestroyDescriptorPool(
			application_pointer->device, application_pointer->descriptor_pool, NULL
		);
		application_pointer->descriptor_pool = VK_NULL_HANDLE;
	}

	if(	application_pointer->descriptor_sets_pointer != NULL ) {
		free( application_pointer->descriptor_sets_pointer );
		application_pointer->descriptor_sets_pointer = NULL;
	}

	if(	application_pointer->descriptor_set_layout != VK_NULL_HANDLE ) {
		vkDestroyDescriptorSetLayout(
			application_pointer->device, application_pointer->descriptor_set_layout, NULL
		);
		application_pointer->descriptor_set_layout = VK_NULL_HANDLE;
	}

	if(	application_pointer->buffer_vertex != VK_NULL_HANDLE ) {
		vkDestroyBuffer(application_pointer->device, application_pointer->buffer_vertex, NULL);
		application_pointer->buffer_vertex = VK_NULL_HANDLE;
	}
	if(	application_pointer->buffer_memory_vertex != VK_NULL_HANDLE ) {
		vkFreeMemory(
			application_pointer->device, application_pointer->buffer_memory_vertex, NULL
		);
		application_pointer->buffer_memory_vertex = VK_NULL_HANDLE;
	}

	if(	application_pointer->buffer_index != VK_NULL_HANDLE ) {
		vkDestroyBuffer( application_pointer->device, application_pointer->buffer_index, NULL );
		application_pointer->buffer_index = VK_NULL_HANDLE;
	}
	if(	application_pointer->buffer_memory_index != VK_NULL_HANDLE ) {
		vkFreeMemory(
			application_pointer->device, application_pointer->buffer_memory_index, NULL
		);
		application_pointer->buffer_memory_index = VK_NULL_HANDLE;
	}

	if(	application_pointer->graphics_pipeline != VK_NULL_HANDLE ) {
		vkDestroyPipeline(
			application_pointer->device, application_pointer->graphics_pipeline, NULL
		);
		application_pointer->graphics_pipeline = VK_NULL_HANDLE;
	}
	if(	application_pointer->pipeline_layout != VK_NULL_HANDLE ) {
		vkDestroyPipelineLayout(
			application_pointer->device, application_pointer->pipeline_layout, NULL
		);
		application_pointer->pipeline_layout = VK_NULL_HANDLE;
	}
	if(	application_pointer->render_pass != VK_NULL_HANDLE ) {
		vkDestroyRenderPass(
			application_pointer->device, application_pointer->render_pass, NULL
		);
		application_pointer->render_pass = VK_NULL_HANDLE;
	}

	if(	application_pointer->synchronization_frame_pointer != NULL )
		vsr_synchronization_frames_destroy(
			application_pointer, application_pointer->frames_in_flight_limit
		);

	if(	application_pointer->render_finished_semaphores_pointer != NULL )
		vsr_synchronization_semaphores_render_finished_destroy(
			application_pointer->device, application_pointer->render_finished_semaphores_pointer,
			application_pointer->render_finished_semaphores_amount
		);

	if(	application_pointer->present_fences_pointer != NULL )
		vsr_synchronization_fence_present_destroy(
			application_pointer->device, application_pointer->present_fences_pointer,
			application_pointer->frames_in_flight_limit
		);

	if(	application_pointer->command_pool_graphic != VK_NULL_HANDLE ) {
		vkDestroyCommandPool(
			application_pointer->device, application_pointer->command_pool_graphic, NULL
		);
		application_pointer->command_pool_graphic = VK_NULL_HANDLE;
	}
	if(	application_pointer->command_pool_transfer != VK_NULL_HANDLE ) {
		vkDestroyCommandPool(
			application_pointer->device, application_pointer->command_pool_transfer, NULL
		);
		application_pointer->command_pool_transfer = VK_NULL_HANDLE;
	}

	vkDestroyDevice( application_pointer->device, NULL );
	application_pointer->device = VK_NULL_HANDLE;
}

static void vsr_swap_chain_cleanup( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	if(	application_pointer->swap_chain_data.frame_buffer			!= VK_NULL_HANDLE ||
		application_pointer->swap_chain_data.frame_buffers_pointer	!= NULL )
	{
		vsr_frame_buffer_destroy(
			application_pointer->device,
			application_pointer->swap_chain_data.frame_buffer,
			application_pointer->swap_chain_data.frame_buffers_pointer,
			application_pointer->swap_chain_data.image_views_amount
		);
		application_pointer->swap_chain_data.frame_buffer			= VK_NULL_HANDLE;
		application_pointer->swap_chain_data.frame_buffers_pointer	= NULL;
	}

	if(	application_pointer->swap_chain_image_views_pointer != NULL ) {
		vsr_image_views_destroy(
			application_pointer->device, application_pointer->swap_chain_image_views_pointer,
			application_pointer->swap_chain_data.image_views_amount
		);
		application_pointer->swap_chain_image_views_pointer = NULL;
	}

	if(	application_pointer->swap_chain_data.swap_chain != VK_NULL_HANDLE ) {
		vkDestroySwapchainKHR(
			application_pointer->device, application_pointer->swap_chain_data.swap_chain, NULL
		);
		application_pointer->swap_chain_data.swap_chain = VK_NULL_HANDLE;
	}
}

static void vsr_application_cleanup( void * restrict argument_pointer ) {
	struct VSR_Application * application_pointer = (struct VSR_Application *) argument_pointer;
	assert_m( application_pointer != NULL, "No application found" );

	if(	application_pointer->is_render_thread_created == true ) {
		pthread_mutex_lock( &application_pointer->render_mutex );
		application_pointer->is_running = false;
		pthread_cond_broadcast( &application_pointer->render_condition );
		pthread_mutex_unlock( &application_pointer->render_mutex );

		pthread_join( application_pointer->render_thread, NULL );
		application_pointer->is_render_thread_created = false;
	}

	vsr_device_and_resources_destroy( application_pointer );

	if(	application_pointer->instance != VK_NULL_HANDLE ) {

#ifndef NDEBUG

		if(	application_pointer->is_debug_messenger_established == true )
			vsr_debug_utils_messenger_extension_destroy(
				application_pointer->instance, application_pointer->debug_messenger_function

			);

#endif

		if(	application_pointer->surface != VK_NULL_HANDLE ) {
			vkDestroySurfaceKHR(
				application_pointer->instance, application_pointer->surface, NULL
			);
			application_pointer->surface = NULL;
		}
		vkDestroyInstance( application_pointer->instance, NULL );
		application_pointer->instance = NULL;
	}
	if(	application_pointer->window_pointer != NULL ) {
		glfwDestroyWindow( application_pointer->window_pointer );
		application_pointer->window_pointer = NULL;
	}
	if(	application_pointer->is_initialized_glfw == true )
		glfwTerminate();

	if(	application_pointer->is_thread_objects_created == true ) {
		application_pointer->is_thread_objects_created = false;
		pthread_cond_destroy( &application_pointer->render_condition );
		pthread_mutex_destroy(&application_pointer->render_mutex );
	}

	bool message_have_to_be_freed;
	for ( char * error_message_pointer;
			(error_message_pointer = woem_pop(&message_have_to_be_freed)) != NULL; )
	{
		fprintf( stderr, "error: %s\n", error_message_pointer );
		if(	message_have_to_be_freed == true )
			free( error_message_pointer );
	}
}

static inline struct VSR_Deletion_Entity vsr_get_delay_deletion_entity_data(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	return (struct VSR_Deletion_Entity) {
		.swap_chain							= application_pointer->swap_chain_data.swap_chain,
		.swap_chain_frame_buffer			= application_pointer->swap_chain_data.frame_buffer,
		.swap_chain_frame_buffer_pointer	= application_pointer->swap_chain_data.
			frame_buffers_pointer,
		.image_views_amount					= application_pointer->swap_chain_data.
			image_views_amount,
		.swap_chain_image_views_pointer		= application_pointer->swap_chain_image_views_pointer,
		.render_finished_semaphores_pointer =
			(application_pointer->capabilities_device.has_swapchain_maintenance_1 == false)
				? application_pointer->render_finished_semaphores_pointer
				: NULL,
		.delete_frame						=
			application_pointer->frame_counter + application_pointer->frames_in_flight_limit
	};
}

static const char * vsr_delay_deletion_cleanup(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	if(	rb_ring_buffer_is_full( &application_pointer->deletion_queue ) == true ) {
		vkDeviceWaitIdle( application_pointer->device );

		while ( rb_ring_buffer_peek(&application_pointer->deletion_queue) != NULL ) {
			struct VSR_Deletion_Entity * entity_pointer = rb_ring_buffer_pop(
				&application_pointer->deletion_queue
			);
			vsr_deletion_entity_destroy( application_pointer, entity_pointer );
		}

		application_pointer->frame_counter = 0;
	}

	struct VSR_Deletion_Entity deletion = vsr_get_delay_deletion_entity_data(application_pointer);

	if(	rb_ring_buffer_push( &application_pointer->deletion_queue, &deletion ) == false )
		return "(vsr_delay_deletion_cleanup) failed to push to deletion queue";

	return NULL;
}

static void vsr_deletion_entity_destroy(
		struct VSR_Application * restrict application_pointer,
		struct VSR_Deletion_Entity * entity_pointer
	)
{
	assert_m( application_pointer	!= NULL, "No application found" );
	assert_m( entity_pointer		!= NULL, "No application found" );

	vsr_frame_buffer_destroy(
		application_pointer->device, entity_pointer->swap_chain_frame_buffer,
		entity_pointer->swap_chain_frame_buffer_pointer, entity_pointer->image_views_amount
	);

	vsr_image_views_destroy(
		application_pointer->device, entity_pointer->swap_chain_image_views_pointer,
		entity_pointer->image_views_amount
	);

	vkDestroySwapchainKHR(application_pointer->device, entity_pointer->swap_chain, NULL);


	if(	entity_pointer->render_finished_semaphores_pointer != NULL )
		vsr_synchronization_semaphores_render_finished_destroy(
			application_pointer->device,
			entity_pointer->render_finished_semaphores_pointer, entity_pointer->image_views_amount
		);
}

static void vsr_image_views_destroy(
		VkDevice device, VkImageView * restrict image_views_pointer, uint32_t image_views_amount
	)
{
	assert_m( image_views_pointer != NULL, "No image views found" );

	for(uint32_t image_view_index = 0; image_view_index < image_views_amount; ++image_view_index)
		vkDestroyImageView( device, image_views_pointer[image_view_index], NULL );

	free( image_views_pointer );
}

static void vsr_delay_deletion_process( struct VSR_Application * restrict application_pointer )
{
	assert_m( application_pointer != NULL, "No application found" );

	for(size_t entity_index = application_pointer->deletion_queue.amount;
			entity_index > 0; --entity_index )
	{
		struct VSR_Deletion_Entity * entity_pointer = rb_ring_buffer_peek(
			&application_pointer->deletion_queue
		);
		if(	entity_pointer == NULL )
			break;

		if(	vsr_frame_reached(
				application_pointer->frame_counter, entity_pointer->delete_frame
			) == false )
			break;

		vsr_deletion_entity_destroy( application_pointer, entity_pointer );
		rb_ring_buffer_discard( &application_pointer->deletion_queue );
	}

	if(	application_pointer->deletion_queue.amount == 0 )
		application_pointer->frame_counter = 0;
	else ++application_pointer->frame_counter;
}

static inline bool vsr_frame_reached(uint32_t frame_current, uint32_t frame_target) {
	return ((uint32_t)frame_current - (uint32_t)frame_target) < (UINT32_MAX / 2);
}

static const char * vsr_descriptor_sets_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );
	assert_m(
		application_pointer->frames_in_flight_limit <= VSR_LIMIT_FRAMES_IN_FLIGHT,
		"Frames in flight limit exceed static array size"
	);

	VkDescriptorSetLayout layouts_array[VSR_LIMIT_FRAMES_IN_FLIGHT];
	for ( uint8_t frame_index = 0;
			frame_index < application_pointer->frames_in_flight_limit; ++frame_index )
		layouts_array[frame_index] = application_pointer->descriptor_set_layout;

	struct VkDescriptorSetAllocateInfo allocate_information = {
		.sType				= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool		= application_pointer->descriptor_pool,
		.descriptorSetCount	= (uint32_t)( application_pointer->frames_in_flight_limit ),
		.pSetLayouts		= layouts_array
	};

	if(	sa_malloc_array(
			&application_pointer->descriptor_sets_pointer,
			application_pointer->frames_in_flight_limit, sizeof(VkDescriptorSet)
		) == false )
		return "(vsr_descriptor_sets_create) allocation size overflow";
	else if ( application_pointer->descriptor_sets_pointer == NULL )
		return "(vsr_descriptor_sets_create) descriptor sets allocation failed";

	if(vkAllocateDescriptorSets(
			application_pointer->device, &allocate_information,
			application_pointer->descriptor_sets_pointer
		) != VK_SUCCESS )
		return "(vsr_descriptor_sets_create) descriptor sets VRAM allocation failed";

	for ( uint8_t frame_index = 0;
			frame_index < application_pointer->frames_in_flight_limit; ++frame_index )
	{
		struct VkDescriptorBufferInfo buffer_create_information = {
			.buffer	= application_pointer->buffers_uniform_pointer[frame_index],
			.range	= sizeof(struct VSR_Uniform_Buffer_Object)
		};

		struct VkWriteDescriptorSet descriptor_write = {
			.sType				= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet				= application_pointer->descriptor_sets_pointer[frame_index],
			.descriptorCount	= 1,
			.descriptorType		= VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo		= &buffer_create_information,
		};

		vkUpdateDescriptorSets( application_pointer->device, 1, &descriptor_write, 0, NULL );
	}

	return NULL;
}

static const char * vsr_descriptor_pool_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	struct VkDescriptorPoolSize pool_size = {
		.type				= VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount	= (uint32_t)(application_pointer->frames_in_flight_limit)
	};

	struct VkDescriptorPoolCreateInfo pool_information = {
		.sType			= VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets		= (uint32_t)(application_pointer->frames_in_flight_limit),
		.poolSizeCount	= 1,
		.pPoolSizes		= &pool_size
	};

	return( vkCreateDescriptorPool_wrapped(
				application_pointer->device, &pool_information, NULL,
				&application_pointer->descriptor_pool
			) == VK_SUCCESS )
		? NULL
		: "(vsr_descriptor_pool_create) descriptor pool creation failed";
}

static const char * vsr_descriptor_set_layout_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	struct VkDescriptorSetLayoutBinding buffer_uniform_object_layout_binding = {
		.descriptorType		= VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount	= 1,
		.stageFlags			= VK_SHADER_STAGE_VERTEX_BIT,
	};

	struct VkDescriptorSetLayoutCreateInfo layout_create_information = {
		.sType			= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount	= 1,
		.pBindings		= &buffer_uniform_object_layout_binding
	};

	return( vkCreateDescriptorSetLayout_wrapped(
				application_pointer->device, &layout_create_information, NULL,
				&application_pointer->descriptor_set_layout
			) == VK_SUCCESS )
		? NULL
		: "(vsr_descriptor_set_layout_create) descriptor set layout creation failed";
}

static const char * vsr_buffer_uniform_create(
		struct VSR_Application * restrict application_pointer,
		const VkPhysicalDeviceProperties * restrict device_properties_pointer
	)
{
	VkDeviceSize buffer_size;
	VkDeviceSize safe_alignment_required =
		(device_properties_pointer->limits.minUniformBufferOffsetAlignment == 0)
		? 1
		: device_properties_pointer->limits.minUniformBufferOffsetAlignment;
	if(	sa_ovf_round_up_uint64_t(
			application_pointer->buffer_uniform_size, safe_alignment_required, &buffer_size
		) == true)
		return "(vsr_buffer_uniform_create) uniform buffer alignment calculus failed";

	VkBuffer * uniform_buffers_pointer;
	if(	sa_malloc_array(
			&uniform_buffers_pointer,
			application_pointer->frames_in_flight_limit,
			sizeof(*uniform_buffers_pointer)
		) == false )
		return "(vsr_buffer_uniform_create) allocation size overflow";
	else if ( uniform_buffers_pointer == NULL )
		return "(vsr_buffer_uniform_create) allocation of uniform buffers failed";

	struct VkBufferCreateInfo buffer_create_information = {
		.sType			= VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size			= buffer_size,
		.usage			= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode	= VK_SHARING_MODE_EXCLUSIVE
	};

	for(uint8_t buffer_uniform_index = 0;
			buffer_uniform_index < application_pointer->frames_in_flight_limit;
			++buffer_uniform_index )
	{
		if(	vkCreateBuffer(
				application_pointer->device, &buffer_create_information, NULL,
				&uniform_buffers_pointer[buffer_uniform_index]
			) != VK_SUCCESS )
		{
			for(uint8_t cleanup_index = 0; cleanup_index < buffer_uniform_index; ++cleanup_index )
			{
				vkDestroyBuffer(
					application_pointer->device, uniform_buffers_pointer[cleanup_index], NULL
				);
			}
			free( uniform_buffers_pointer );
			return "(vsr_buffer_uniform_create) buffer creation failed";
		}
	}
	application_pointer->buffers_uniform_pointer = uniform_buffers_pointer;

	return NULL;
}

static const char * vsr_buffer_uniforms_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	application_pointer->buffer_uniform_size = sizeof( struct VSR_Uniform_Buffer_Object );

	VkPhysicalDeviceProperties device_properties;
	vkGetPhysicalDeviceProperties(application_pointer->device_physical, &device_properties);

	const char * error_message_pointer;
	if((error_message_pointer = vsr_buffer_uniform_create(
			application_pointer, &device_properties
		)) != NULL )
		return error_message_pointer;

	VkMemoryRequirements memory_requirements;
	vkGetBufferMemoryRequirements(
		application_pointer->device, application_pointer->buffers_uniform_pointer[0],
		&memory_requirements
	);

	const struct VSR_Memory_Properties memory_properties[] = {
		{
			.list_required	=
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			.list_forbidden	= 0
		}, {
			.list_required	=
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			.list_forbidden	= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		}, {
			.list_required	= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			.list_forbidden	= 0
		}
	};

	const struct VSR_Memory_Levels_Requirements memory_requirements_list = {
		.memory_properties = memory_properties,
		.properties_amount = (uint32_t) sizeof(memory_properties) / sizeof(*memory_properties)
	};

	uint32_t memory_type_index;
	if(	vsr_memory_type_find(
			application_pointer, memory_requirements.memoryTypeBits, &memory_requirements_list,
			&memory_type_index
		) == false )
		return "(vsr_buffer_uniforms_create) suitable memory type not found";

	application_pointer->is_buffer_uniform_coherent =
		vsr_memory_type_is_coherent(application_pointer, memory_type_index);

	VkDeviceSize atom_size = 1;
	if(	application_pointer->is_buffer_uniform_coherent	== false &&
		device_properties.limits.nonCoherentAtomSize	!= 0 )
		atom_size = device_properties.limits.nonCoherentAtomSize;

	VkDeviceSize alignment_base = (memory_requirements.alignment == 0)
		? 1
		: memory_requirements.alignment;
	if(	sa_ovf_round_up_uint64_t( alignment_base, atom_size, &alignment_base ) == true )
		return "(vsr_buffer_uniforms_create) alignment overflow";
	if(	sa_ovf_round_up_uint64_t(
			memory_requirements.size, alignment_base,
			&application_pointer->buffer_uniform_size_alignment
		) == true )
		return "(vsr_buffer_uniforms_create) allocation size alignment overflow";

	VkDeviceSize allocation_size;
	if(	sa_ovf_mul_uint64_t(
			application_pointer->buffer_uniform_size_alignment,
			application_pointer->frames_in_flight_limit, &allocation_size
		) == true )
		return "(vsr_buffer_uniforms_create) invalid allocation size";

	if(	sa_ovf_round_up_uint64_t(
			application_pointer->buffer_uniform_size, atom_size,
			&application_pointer->buffer_uniform_size_flush
		) == true )
		return "(vsr_buffer_uniforms_create) flush size alignment overflow";

	struct VkMemoryAllocateInfo allocate_information = {
		.sType				= VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize		= allocation_size,
		.memoryTypeIndex	= memory_type_index
	};

	if(	vkAllocateMemory_wrapped(
			application_pointer->device, &allocate_information, NULL,
			&application_pointer->buffers_uniform_memory
		) != VK_SUCCESS )
		return "(vsr_buffer_uniforms_create) uniform buffer memory allocation failed";

	if(	vkMapMemory(
			application_pointer->device, application_pointer->buffers_uniform_memory, 0,
			VK_WHOLE_SIZE, 0, &application_pointer->buffers_uniform_mapped_pointer
		) != VK_SUCCESS )
		return "(vsr_buffer_uniforms_create) memory mapping failed";

	for ( uint8_t buffer_uniform_index = 0;
			buffer_uniform_index < application_pointer->frames_in_flight_limit;
			++buffer_uniform_index )
	{
		VkDeviceSize size_offset;
		if(	sa_ovf_mul_uint64_t(
				buffer_uniform_index, application_pointer->buffer_uniform_size_alignment,
				&size_offset
			) == true )
			return "(vsr_buffer_uniforms_create) invalid offset size";
		if(	vkBindBufferMemory(
				application_pointer->device,
				application_pointer->buffers_uniform_pointer[buffer_uniform_index],
				application_pointer->buffers_uniform_memory, size_offset
			) != VK_SUCCESS )
			return "(vsr_buffer_uniforms_create) binding buffer memory failed";
	}

	return NULL;
}

static inline bool vsr_memory_type_is_coherent(
		const struct VSR_Application * restrict application_pointer,
		const uint32_t memory_type_index
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	return
		(application_pointer->memory_properties.memoryTypes[memory_type_index].propertyFlags
			& VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
}

static const char * vsr_buffers_fast_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	const char							* error_message_pointer;
	struct VSR_Buffer_Allocation_Data	allocation_data;
	VkCommandPool						command_pool;
	const struct VSR_Vertex				vertices_array[] = {
		{ {-0.5f, -0.5f}, { 1.0f, 0.0f, 0.0f} },
		{ { 0.5f, -0.5f}, { 0.0f, 1.0f, 0.0f} },
		{ { 0.5f,  0.5f}, { 0.0f, 0.0f, 1.0f} },
		{ {-0.5f,  0.5f}, { 1.0f, 1.0f, 1.0f} }
	};

	const VkDeviceSize staging_union_size = sizeof(vertices_array) > sizeof(global_indices_array)
		? sizeof(vertices_array)
		: sizeof(global_indices_array);

	if((error_message_pointer = vsr_buffer_staging_create(
			application_pointer, staging_union_size, &allocation_data
		)) != NULL )
		return error_message_pointer;

	if((error_message_pointer = vsr_inclusive_command_pool_create(
			application_pointer, &command_pool, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			application_pointer->queue_family_indices.transfer_family
		)) != NULL )
		goto cleanup;

	struct VSR_Buffer_Allocation_Data allocation_data_vertex;
	if((error_message_pointer = vsr_buffer_inclusive_create(
			application_pointer, vertices_array, sizeof(vertices_array),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, allocation_data, command_pool,
			&allocation_data_vertex
		)) != NULL )
		goto cleanup_pool;

	struct VSR_Buffer_Allocation_Data allocation_data_index;
	if((error_message_pointer = vsr_buffer_inclusive_create(
			application_pointer, global_indices_array, sizeof(global_indices_array),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT, allocation_data, command_pool,
			&allocation_data_index
		)) != NULL )
		goto cleanup_buffer;

	application_pointer->buffer_vertex			= allocation_data_vertex.buffer;
	application_pointer->buffer_memory_vertex	= allocation_data_vertex.memory;
	application_pointer->buffer_index			= allocation_data_index.buffer;
	application_pointer->buffer_memory_index	= allocation_data_index.memory;

	vkDestroyCommandPool( application_pointer->device, command_pool, NULL );
	vkDestroyBuffer( application_pointer->device, allocation_data.buffer, NULL );
	vkFreeMemory( application_pointer->device, allocation_data.memory, NULL );

	return NULL;

cleanup_buffer:
	vkDestroyBuffer( application_pointer->device, allocation_data_vertex.buffer, NULL );
	vkFreeMemory( application_pointer->device, allocation_data_vertex.memory, NULL );

cleanup_pool:
	vkDestroyCommandPool( application_pointer->device, command_pool, NULL );

cleanup:
	vkDestroyBuffer( application_pointer->device, allocation_data.buffer, NULL );
	vkFreeMemory( application_pointer->device, allocation_data.memory, NULL );
	return error_message_pointer;
}

static inline const char * vsr_buffer_staging_create(
		struct VSR_Application * restrict application_pointer, VkDeviceSize buffer_size,
		struct VSR_Buffer_Allocation_Data * restrict out_allocation_data_pointer
	)
{
	assert_m( application_pointer			!= NULL, "No application found"				);
	assert_m( out_allocation_data_pointer	!= NULL, "No allocation data storage found"	);

	const struct VSR_Memory_Properties memory_properties[] = {
		{
			.list_required	= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			.list_forbidden	=
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		}, {
			.list_required	= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			.list_forbidden	= 0
		}
	};

	const struct VSR_Memory_Levels_Requirements memory_requirements_list = {
		.memory_properties = memory_properties,
		.properties_amount = (uint32_t) sizeof(memory_properties) / sizeof(*memory_properties)
	};

	return vsr_buffer_create(
		application_pointer, buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		&memory_requirements_list, out_allocation_data_pointer
	);
}

static const char * vsr_buffer_inclusive_create(
		struct VSR_Application * restrict application_pointer,
		const void * restrict buffer_pointer, VkDeviceSize buffer_size,
		VkBufferUsageFlags buffer_usage, struct VSR_Buffer_Allocation_Data allocation_data_staging,
		VkCommandPool command_pool,
		struct VSR_Buffer_Allocation_Data * restrict out_allocation_data_pointer
	)
{
	assert_m( application_pointer			!= NULL, "No application found"				);
	assert_m( buffer_pointer				!= NULL, "No source buffer found"			);
	assert_m( out_allocation_data_pointer	!= NULL, "No allocation data storage found"	);

	void * data_pointer;
	if(	vkMapMemory(
			application_pointer->device, allocation_data_staging.memory, 0, VK_WHOLE_SIZE, 0,
			&data_pointer
		) != VK_SUCCESS )
		return "(vsr_buffer_inclusive_create) staging buffer memory mapping failed";

	memcpy( data_pointer, buffer_pointer, (size_t) buffer_size );

	if(	allocation_data_staging.is_coherent == false ) {
		const struct VkMappedMemoryRange flush_range = {
			.sType	= VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
			.memory	= allocation_data_staging.memory,
			.size	= VK_WHOLE_SIZE
		};
		if(	vkFlushMappedMemoryRanges(
				application_pointer->device, 1, &flush_range
			) != VK_SUCCESS)
		{
			vkUnmapMemory( application_pointer->device, allocation_data_staging.memory );
			return "(vsr_buffer_inclusive_create) staging buffer flushing failed";
		}
	}

	vkUnmapMemory( application_pointer->device, allocation_data_staging.memory );

	const struct VSR_Memory_Properties memory_properties[] = {
		{
			.list_required	=
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			.list_forbidden	= 0
		}, {
			.list_required	= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			.list_forbidden	= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		}, {
			.list_required	= 0,
			.list_forbidden	= 0
		}
	};

	const struct VSR_Memory_Levels_Requirements memory_requirements_list = {
		.memory_properties = memory_properties,
		.properties_amount = (uint32_t) sizeof(memory_properties) / sizeof(*memory_properties)
	};

	const char * error_message_pointer;
	if((error_message_pointer = vsr_buffer_create(
			application_pointer, buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | buffer_usage,
			&memory_requirements_list, out_allocation_data_pointer
		)) != NULL )
		return error_message_pointer;

	if((error_message_pointer = vsr_buffer_copy(
			application_pointer, allocation_data_staging.buffer,
			out_allocation_data_pointer->buffer, buffer_size, command_pool
		)) != NULL )
	{
		vkDestroyBuffer(application_pointer->device, out_allocation_data_pointer->buffer, NULL);
		vkFreeMemory( application_pointer->device, out_allocation_data_pointer->memory, NULL );
	}

	return error_message_pointer;
}

static const char * vsr_buffer_copy(
		struct VSR_Application * restrict application_pointer, VkBuffer buffer_source,
		VkBuffer buffer_destination, VkDeviceSize size, VkCommandPool command_pool
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	const char * error_message_pointer = NULL;
	struct VkCommandBufferAllocateInfo allocate_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool		= command_pool,
		.level				= VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount	= 1
	};

	VkCommandBuffer command_buffer;
	if(vkAllocateCommandBuffers(
			application_pointer->device, &allocate_information, &command_buffer
		) != VK_SUCCESS )
		return "(vsr_buffer_copy) command buffer allocation failed";

	struct VkCommandBufferBeginInfo command_buffer_begin_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		/* type of command buffer using */
		.flags				= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	if(	vkBeginCommandBuffer(command_buffer, &command_buffer_begin_information) != VK_SUCCESS ) {
		error_message_pointer = "(vsr_buffer_copy) recording command buffer beginning failed";
		goto out;
	}

	struct VkBufferCopy region_to_copy = (struct VkBufferCopy) { .size = size };
	vkCmdCopyBuffer( command_buffer, buffer_source, buffer_destination, 1, &region_to_copy );

	vsr_buffer_copy_barrier(
		command_buffer, buffer_destination, size,
		application_pointer->queue_family_indices.graphics_family,
		application_pointer->queue_family_indices.transfer_family
	);

	if(	vkEndCommandBuffer( command_buffer ) != VK_SUCCESS ) {
		error_message_pointer = "(vsr_buffer_copy) command buffer recording failed";
		goto out;
	}

	struct VkSubmitInfo submit_information = (struct VkSubmitInfo) {
		.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount	= 1,
		.pCommandBuffers	= &command_buffer
	};

	if(	vkQueueSubmit(
			application_pointer->transfer_queue, 1, &submit_information, VK_NULL_HANDLE
		) != VK_SUCCESS )
	{
		error_message_pointer = "(vsr_buffer_copy) copy command buffer submission failed";
		goto out;
	}
	if(	vkQueueWaitIdle( application_pointer->transfer_queue ) != VK_SUCCESS ) {
		error_message_pointer = "(vsr_buffer_copy) transfer queue wait failed";
		goto out;
	}

	vsr_buffer_ownership_acquire(application_pointer, buffer_destination, size);

out:
	vkFreeCommandBuffers( application_pointer->device, command_pool, 1, &command_buffer );
	return error_message_pointer;
}

static const char * vsr_buffer_ownership_acquire(
		struct VSR_Application * restrict application_pointer,
		VkBuffer buffer_destination, VkDeviceSize size
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	if(	application_pointer->queue_family_indices.transfer_family ==
		application_pointer->queue_family_indices.graphics_family )
		return NULL;

	struct VkCommandBufferAllocateInfo allocation_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool		= application_pointer->command_pool_graphic,
		.level				= VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount	= 1
	};
	VkCommandBuffer command_buffer;
	if(	vkAllocateCommandBuffers(
			application_pointer->device, &allocation_information, &command_buffer
		) != VK_SUCCESS )
		return "(vsr_buffer_ownership_acquire) command buffer allocation failed";

	const char * error_message_pointer = vsr_buffer_ownership_acquire_record(
		application_pointer, command_buffer, buffer_destination, size
	);
	if(	error_message_pointer != NULL ) {
		vkFreeCommandBuffers(
			application_pointer->device, application_pointer->command_pool_graphic, 1,
			&command_buffer
		);
		return error_message_pointer;
	}

	return vsr_buffer_ownership_acquire_submit_wait(application_pointer, command_buffer);
}

static const char * vsr_buffer_ownership_acquire_record(
		struct VSR_Application * restrict application_pointer,
		VkCommandBuffer command_buffer, VkBuffer buffer_destination, VkDeviceSize size
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	struct VkCommandBufferBeginInfo command_buffer_begin_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		/* type of command buffer using */
		.flags				= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	if(	vkBeginCommandBuffer(command_buffer, &command_buffer_begin_information) != VK_SUCCESS )
		return "(vsr_buffer_ownership_acquire_record) recording command buffer beginning failed";

	struct VkBufferMemoryBarrier memory_barrier_acquire = {
		.sType				= VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.dstAccessMask		= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
		.srcQueueFamilyIndex= application_pointer->queue_family_indices.transfer_family,
		.dstQueueFamilyIndex= application_pointer->queue_family_indices.graphics_family,
		.buffer				= buffer_destination,
		.size				= size
	};
	vkCmdPipelineBarrier(
		command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
		0, 0, NULL, 1, &memory_barrier_acquire, 0, NULL
	);

	return (vkEndCommandBuffer( command_buffer ) != VK_SUCCESS)
		? "(vsr_buffer_ownership_acquire_record) command buffer recording failed"
		: NULL;
}

static const char * vsr_buffer_ownership_acquire_submit_wait(
		struct VSR_Application * restrict application_pointer, VkCommandBuffer command_buffer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	struct VkSubmitInfo submit_information = {
		.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount	= 1,
		.pCommandBuffers	= &command_buffer
	};

	VkResult result = vkQueueSubmit(
		application_pointer->graphics_queue, 1, &submit_information, VK_NULL_HANDLE
	);

	const char * error_message_pointer = NULL;
	if(	result == VK_SUCCESS )
		result = vkQueueWaitIdle(application_pointer->graphics_queue);
	else
		error_message_pointer =
			"(vsr_buffer_ownership_acquire_submit_wait) acquiring submission failed";

	vkFreeCommandBuffers(
		application_pointer->device, application_pointer->command_pool_graphic, 1, &command_buffer
	);

	return (error_message_pointer != NULL)
		? error_message_pointer
		: ( result != VK_SUCCESS )
			? "(vsr_buffer_ownership_acquire_submit_wait) graphics queue wait failed"
			: NULL;
}

static void vsr_buffer_copy_barrier(
		VkCommandBuffer command_buffer, VkBuffer buffer_destination, VkDeviceSize size,
		uint32_t graphics_family, uint32_t transfer_family
	)
{
	if(	graphics_family == transfer_family )
		return;

	struct VkBufferMemoryBarrier memory_barrier_release = {
		.sType				= VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask		= VK_ACCESS_TRANSFER_WRITE_BIT,
		.srcQueueFamilyIndex= transfer_family,
		.dstQueueFamilyIndex= graphics_family,
		.buffer				= buffer_destination,
		.size				= size
	};
	vkCmdPipelineBarrier(
		command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		0, 0, NULL, 1, &memory_barrier_release, 0, NULL
	);
}

static const char * vsr_buffer_create(
		struct VSR_Application * restrict application_pointer, VkDeviceSize size,
		VkBufferUsageFlags usage,
		const struct VSR_Memory_Levels_Requirements * restrict requirements_list,
		struct VSR_Buffer_Allocation_Data * restrict out_allocation_data_pointer
	)
{
	assert_m( requirements_list				!= NULL, "No requirements found"			);
	assert_m( application_pointer			!= NULL, "No application found"				);
	assert_m( out_allocation_data_pointer	!= NULL, "No allocation data storage found"	);

	struct VkBufferCreateInfo buffer_create_information = {
		.sType			= VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size			= size,
		.usage			= usage,
		.sharingMode	= VK_SHARING_MODE_EXCLUSIVE
	};

	if(	vkCreateBuffer(
			application_pointer->device, &buffer_create_information, NULL,
			&out_allocation_data_pointer->buffer
		) != VK_SUCCESS )
		return "(vsr_buffer_create) buffer creation failed";

	VkMemoryRequirements memory_requirements;
	vkGetBufferMemoryRequirements(
		application_pointer->device, out_allocation_data_pointer->buffer, &memory_requirements
	);

	const char * error_message_pointer;
	uint32_t memory_type_index = 0;
	if(	vsr_memory_type_find(
			application_pointer, memory_requirements.memoryTypeBits, requirements_list,
			&memory_type_index
		) == false )
	{
		error_message_pointer = "(vsr_buffer_create) suitable memory type not found";
		goto cleanup;
	}

	out_allocation_data_pointer->is_coherent =
		vsr_memory_type_is_coherent(application_pointer, memory_type_index);

	struct VkMemoryAllocateInfo allocate_information = (struct VkMemoryAllocateInfo) {
		.sType				= VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize		= memory_requirements.size,
		.memoryTypeIndex	= memory_type_index
	};

	if(vkAllocateMemory(
			application_pointer->device, &allocate_information, NULL,
			&out_allocation_data_pointer->memory
		) != VK_SUCCESS )
	{
		error_message_pointer = "(vsr_buffer_create) buffer memory allocation failed";
		goto cleanup;
	}

	if(	vkBindBufferMemory(
			application_pointer->device, out_allocation_data_pointer->buffer,
			out_allocation_data_pointer->memory, 0
		) != VK_SUCCESS )
	{
		error_message_pointer = "(vsr_buffer_create) buffer memory binding failed";
		goto cleanup_memory;
	}

	return NULL;

cleanup_memory:
	vkFreeMemory( application_pointer->device, out_allocation_data_pointer->memory, NULL );
	out_allocation_data_pointer->memory = VK_NULL_HANDLE;

cleanup:
	vkDestroyBuffer( application_pointer->device, out_allocation_data_pointer->buffer, NULL );
	out_allocation_data_pointer->buffer = VK_NULL_HANDLE;
	return error_message_pointer;
}

static bool vsr_memory_type_find(
		struct VSR_Application * restrict application_pointer, uint32_t type_filter,
		const struct VSR_Memory_Levels_Requirements * restrict memory_requirements_pointer,
		uint32_t * restrict out_type_index_pointer
	)
{
	assert_m( application_pointer			!= NULL, "No application found"			);
	assert_m( out_type_index_pointer		!= NULL, "No type index storage found"	);
	assert_m( memory_requirements_pointer	!= NULL, "No memory required list found");

	VkPhysicalDeviceMemoryProperties * memory_properties_pointer =
		&application_pointer->memory_properties;
	VkDeviceSize candidate_size_best = 0;
	uint32_t candidate_best = UINT32_MAX;
	int64_t candidate_priority_best	= -1;

	if(	memory_properties_pointer->memoryTypeCount > VK_MAX_MEMORY_TYPES ) {
		VSR_DEBUG_LOGF(
			"(vsr_memory_type_find) memory type count %u exceeds safe limit",
			memory_properties_pointer->memoryTypeCount
		);
		return false;
	}

	for(uint32_t candidate = 0;
			candidate < memory_properties_pointer->memoryTypeCount; ++candidate )
	{
		if(	(type_filter & ( (uint32_t) 1 << candidate)) == 0 )
			continue;

		VkMemoryPropertyFlags candidate_flags_current =
			memory_properties_pointer->memoryTypes[candidate].propertyFlags;

		uint32_t candidate_priority_current = UINT32_MAX;

		int64_t required_index_worst = (candidate_priority_best != -1)
			? candidate_priority_best
			: memory_requirements_pointer->properties_amount - 1;
		for(int64_t required_index_best = 0;
				required_index_best <= required_index_worst; ++required_index_best)
		{
			struct VSR_Memory_Properties memory_properties_current =
				memory_requirements_pointer->memory_properties[required_index_best];
			if((candidate_flags_current & memory_properties_current.list_required)
				== memory_properties_current.list_required)
			{
				if((candidate_flags_current & memory_properties_current.list_forbidden) == 0) {
					candidate_priority_current = (uint32_t) required_index_best;
					break;
				}
			}
		}

		if(	candidate_priority_current == UINT32_MAX )
			continue;

		uint32_t heap_index = memory_properties_pointer->memoryTypes[candidate].heapIndex;
		VkDeviceSize candidate_size_current =
			memory_properties_pointer->memoryHeaps[heap_index].size;

		if(	candidate_priority_best == -1 ||
			candidate_priority_current < candidate_priority_best ||
			(candidate_priority_current == candidate_priority_best &&
			candidate_size_current > candidate_size_best) )
		{
			candidate_priority_best = candidate_priority_current;
			candidate_size_best = candidate_size_current;
			candidate_best = candidate;
		}
	}

	if(	candidate_priority_best == -1 )
		return false;

	*out_type_index_pointer = (uint32_t) candidate_best;
	return true;
}

static struct VkVertexInputBindingDescription vsr_get_binding_description(void) {
	struct VkVertexInputBindingDescription binding_description = {
		.stride		= sizeof(struct VSR_Vertex),	/* size of entry data in bytes			*/
		.inputRate	= VK_VERTEX_INPUT_RATE_VERTEX	/* move to next data after each vertex	*/
	};
	return binding_description;
}

static bool vsr_swap_chain_is_extent_needs_update(
		struct VSR_Application * restrict application_pointer
	)
{
	VkSurfaceCapabilitiesKHR capabilities;
	if(	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
			application_pointer->device_physical, application_pointer->surface, &capabilities
		) != VK_SUCCESS )
		return true;

	/* under external mutex */
	struct VkExtent2D current_extent = (capabilities.currentExtent.width != UINT32_MAX)
		? capabilities.currentExtent
		: vsr_swap_extent_choose(
			&capabilities,
			(uint32_t) application_pointer->frame_state.width,
			(uint32_t) application_pointer->frame_state.height
		);

	return
		current_extent.width	!= application_pointer->swap_chain_data.extent.width ||
		current_extent.height	!= application_pointer->swap_chain_data.extent.height;
}

static bool vsr_swap_chain_recreate( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	pthread_mutex_lock( &application_pointer->render_mutex );

	bool is_minimized = application_pointer->is_minimized;

	pthread_mutex_unlock( &application_pointer->render_mutex );

	if(	is_minimized == true )
		return false;

	if(	application_pointer->swap_chain_recreate_failed_amount >=
			VSR_LIMIT_FAILURES_SWAPCHAIN_RECREATE )
	{
		vsr_frame_render_failed(
			application_pointer, "(vsr_swap_chain_recreate) swap chain recreation failed"
		);
		return false;
	}

	return vsr_swap_chain_recreate_data( application_pointer );
}

static bool vsr_swap_chain_recreate_data(struct VSR_Application * restrict application_pointer) {
	assert_m( application_pointer != NULL, "No application found" );

	struct VSR_Swap_Chain_Data	new_swap_chain_data;
	VkImage						* swap_chain_images_pointer;
	const char					* error_message_pointer;

	if((error_message_pointer = vsr_swap_chain_create(
			application_pointer, &new_swap_chain_data, &swap_chain_images_pointer
		)) != NULL )
		goto cleanup;

	VkImageView * swap_chain_image_views_pointer_new;
	if((error_message_pointer = vsr_image_views_create(
			application_pointer, swap_chain_images_pointer, new_swap_chain_data.image_views_amount,
			new_swap_chain_data.image_format, &swap_chain_image_views_pointer_new
		)) != NULL )
		goto cleanup_swap_chain;

	if((error_message_pointer = vsr_frame_buffer_create(
			application_pointer, swap_chain_image_views_pointer_new, &new_swap_chain_data
		)) != NULL )
		goto cleanup_image_views;

	if((error_message_pointer = vsr_swap_chain_deletion_resources_handle(
			application_pointer, new_swap_chain_data.image_views_amount
		)) != NULL )
		goto cleanup_frame_buffer;

	free( swap_chain_images_pointer );

	vsr_swap_chain_recreate_data_commit(
		application_pointer, &new_swap_chain_data, swap_chain_image_views_pointer_new
	);

	return true;

cleanup_frame_buffer:
	vsr_frame_buffer_destroy(
		application_pointer->device, new_swap_chain_data.frame_buffer,
		new_swap_chain_data.frame_buffers_pointer, new_swap_chain_data.image_views_amount
	);

cleanup_image_views:
	vsr_image_views_destroy(
		application_pointer->device, swap_chain_image_views_pointer_new,
		new_swap_chain_data.image_views_amount
	);

cleanup_swap_chain:
	vkDestroySwapchainKHR( application_pointer->device, new_swap_chain_data.swap_chain, NULL );
	free( swap_chain_images_pointer );

cleanup:
	++application_pointer->swap_chain_recreate_failed_amount;
	VSR_DEBUG_LOGF( "Error: %s", error_message_pointer );

	return false;
}

static void vsr_swap_chain_recreate_data_commit(
		struct VSR_Application * restrict application_pointer,
		const struct VSR_Swap_Chain_Data * restrict swap_chain_data_pointer,
		VkImageView * restrict swap_chain_image_views_pointer
	)
{
	assert_m( application_pointer			!= NULL, "No application found"				);
	assert_m( swap_chain_data_pointer		!= NULL, "No swap chain data found"			);
	assert_m( swap_chain_image_views_pointer!= NULL, "No swap chain image views found"	);

	application_pointer->swap_chain_data					= *swap_chain_data_pointer;
	application_pointer->swap_chain_image_views_pointer		= swap_chain_image_views_pointer;
	application_pointer->swap_chain_recreate_failed_amount	= 0;

	atomic_store_explicit(
		&application_pointer->swap_chain_extent_packed,
		((uint64_t) swap_chain_data_pointer->extent.width << 32) |
			(uint64_t) swap_chain_data_pointer->extent.height,
		memory_order_relaxed
	);

	atomic_store_explicit(
		&application_pointer->is_swap_chain_valid, true, memory_order_release
	);
}

static const char * vsr_swap_chain_deletion_resources_handle(
		struct VSR_Application * restrict application_pointer, const uint32_t image_views_amount
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	if(	application_pointer->capabilities_device.has_swapchain_maintenance_1 == true )
		return vsr_delay_deletion_cleanup(application_pointer);

	return vsr_swap_chain_render_finished_semaphores_recreate(
		application_pointer, image_views_amount
	);
}

static const char * vsr_swap_chain_render_finished_semaphores_recreate(
		struct VSR_Application * restrict application_pointer, const uint32_t image_views_amount
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	const char * error_message_pointer;
	VkSemaphore * render_finished_semaphores_pointer;
	if((error_message_pointer = vsr_synchronization_semaphores_render_finished_create(
			application_pointer, &render_finished_semaphores_pointer, image_views_amount
		)) != NULL )
		return error_message_pointer;

	if(	vkQueueWaitIdle( application_pointer->present_queue ) != VK_SUCCESS ) {
		vsr_synchronization_semaphores_render_finished_destroy(
			application_pointer->device, render_finished_semaphores_pointer, image_views_amount
		);
		return "(vsr_swap_chain_render_finished_semaphores_recreate) present queue wait failed";
	}

	struct VSR_Deletion_Entity deletion_data =
		vsr_get_delay_deletion_entity_data(application_pointer);
	vsr_deletion_entity_destroy( application_pointer, &deletion_data );

	application_pointer->render_finished_semaphores_pointer	= render_finished_semaphores_pointer;
	application_pointer->render_finished_semaphores_amount	= image_views_amount;

	return NULL;
}

static bool vsr_surface_create( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	if(	glfwCreateWindowSurface(
			application_pointer->instance, application_pointer->window_pointer, NULL,
			&application_pointer->surface
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_surface_create) window surface creation failed" );
		return false;
	}
	return true;
}

static const char * vsr_device_logical_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	uint32_t unique_families_array[VSR_QUEUE_FAMILIES_AMOUNT];
	uint32_t
		families_amount	= 0,
		graphics_family	= application_pointer->queue_family_indices.graphics_family,
		transfer_family	= application_pointer->queue_family_indices.transfer_family,
		present_family	= application_pointer->queue_family_indices.present_family;

	unique_families_array[families_amount++] =
		application_pointer->queue_family_indices.graphics_family;
	if(	present_family != graphics_family )
		unique_families_array[families_amount++] = present_family;

	if(	transfer_family != graphics_family && transfer_family != present_family )
		unique_families_array[families_amount++] = transfer_family;

	struct VkDeviceQueueCreateInfo * queue_create_informations_array;
	if(	sa_malloc_array(
			&queue_create_informations_array, families_amount,
			sizeof(*queue_create_informations_array)
		) == false )
		return "(vsr_device_logical_create) allocation size overflow";
	else if ( queue_create_informations_array == NULL )
		return "(vsr_device_logical_create) queue create information failed to allocate";

	/* priority in [ 0.0f; 1.0f ] */
	const float queue_priority = 1.0f;
	for( size_t queue_family_index = 0;
			queue_family_index < families_amount; ++queue_family_index)
	{
		queue_create_informations_array[queue_family_index] = (struct VkDeviceQueueCreateInfo) {
			.sType				= VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex	= unique_families_array[queue_family_index],
			.queueCount			= 1,
			.pQueuePriorities	= &queue_priority
		};
	}

	VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swap_chains_maintenance1_features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT,
		.swapchainMaintenance1 = VK_TRUE
	};
	VkPhysicalDeviceImagelessFramebufferFeaturesKHR imageless_features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES_KHR,
		.imagelessFramebuffer	= VK_TRUE
	};

	void * features_chain_pointer = NULL;
	if(	application_pointer->capabilities_device.has_swapchain_maintenance_1 == true ) {
		swap_chains_maintenance1_features.pNext	= features_chain_pointer;
		features_chain_pointer					= &swap_chains_maintenance1_features;
	}
	if(	application_pointer->capabilities_device.has_imageless_frame_buffer == true ) {
		imageless_features.pNext				= features_chain_pointer;
		features_chain_pointer					= &imageless_features;
	}

	const char * error_message_pointer = vsr_device_create(
		application_pointer, features_chain_pointer, families_amount,
		queue_create_informations_array
	);
	free( queue_create_informations_array );

	if(	error_message_pointer != NULL )
		return error_message_pointer;

	vkGetDeviceQueue(
		application_pointer->device, application_pointer->queue_family_indices.graphics_family,
		0, &application_pointer->graphics_queue
	);
	vkGetDeviceQueue(
		application_pointer->device, application_pointer->queue_family_indices.present_family,
		0, &application_pointer->present_queue
	);
	if(	application_pointer->queue_family_indices.has_transfer_family == true ) {
		vkGetDeviceQueue(
			application_pointer->device,
			application_pointer->queue_family_indices.transfer_family, 0,
			&application_pointer->transfer_queue
		);
	} else {
		application_pointer->transfer_queue = application_pointer->graphics_queue;
	}

	return NULL;
}

static const char * vsr_device_create(
		struct VSR_Application * restrict application_pointer,
		void * restrict features_chain_pointer, uint32_t families_amount,
		struct VkDeviceQueueCreateInfo * restrict queue_create_informations_array
	)
{
	const char * extensions_enabled_array[VSR_EXTENSIONS_AMOUNT_DEVICE_MAXIMAL];
	struct VSR_Extension_Names_Mutable extensions_enabled = {
		.data_pointer	= extensions_enabled_array,
		.amount			= VSR_EXTENSIONS_AMOUNT_DEVICE_MAXIMAL
	};
	vsr_device_extensions_enabled_array_fill(
		application_pointer, &extensions_enabled
	);
	struct VkDeviceCreateInfo create_information = {
		.sType					= VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext					= features_chain_pointer,
		.queueCreateInfoCount	= families_amount,
		.pQueueCreateInfos		= queue_create_informations_array,
		.enabledExtensionCount	= extensions_enabled.amount,
		.ppEnabledExtensionNames= extensions_enabled.data_pointer
	};

	if(	vkCreateDevice_wrapped(
			application_pointer->device_physical, &create_information, NULL,
			&application_pointer->device
		) != VK_SUCCESS )
		return "(vsr_device_create) logical device failed to create";

	return NULL;
}

static bool vsr_device_physical_select( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	uint32_t devices_amount = 0;
	if(	vkEnumeratePhysicalDevices(
			application_pointer->instance, &devices_amount, NULL
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_device_physical_select) enumeration physical devices failed" );
		return false;
	}

	if(	devices_amount == 0 ) {
		woem_push( "(vsr_device_physical_select) no GPUs with Vulkan support was found" );
		return false;
	}

	VkPhysicalDevice * devices_pointer;
	if(	sa_malloc_array( &devices_pointer, devices_amount, sizeof(*devices_pointer) ) == false ) {
		woem_push( "(vsr_device_physical_select) allocation size overflow" );
		return false;
	} else if ( devices_pointer == NULL ) {
		woem_push( "(vsr_device_physical_select) device memory allocation failed" );
		return false;
	}

	VkResult result = vkEnumeratePhysicalDevices(
		application_pointer->instance, &devices_amount, devices_pointer
	);
	if(	result != VK_SUCCESS && assert_check_mf(
		result != VK_INCOMPLETE, "lack of memory to enumerate %u physical devices", devices_amount
		) == true )
	{
		free( devices_pointer );
		woem_push( "(vsr_device_physical_select) retrieving physical devices failed" );
		return false;
	}

	struct VSR_Device_Candidate candidate_best = {0};
	uint32_t rated_gpus_amount = 0;
	const char * error_message_pointer_last = NULL;
	for ( uint32_t device_current = 0; device_current < devices_amount; ++device_current )
	{
		struct VSR_Device_Candidate candidate_current = {0};
		const char * error_message_pointer_current = vsr_device_suitability_rate(
			application_pointer, devices_pointer[device_current], &candidate_current
		);

		if(	error_message_pointer_current != NULL ) {
			VSR_DEBUG_LOGF(
				"Warning (vsr_device_physical_select): GPU %u skipped: %s",
				device_current, error_message_pointer_current
			);
			error_message_pointer_last = error_message_pointer_current;
			continue;
		}
		++rated_gpus_amount;
		if(	candidate_current.score > candidate_best.score )
			candidate_best = candidate_current;
	}
	free( devices_pointer );

	VSR_DEBUG_LOGF( "\nbest gpu score: %u\n", candidate_best.score );

	if(	candidate_best.score == 0 ) {
		if(	rated_gpus_amount == 0 )
			woem_push(
				"(vsr_device_physical_select) machine broken; %s", error_message_pointer_last
			);
		else
			woem_push( "(vsr_device_physical_select) suitable GPU wasn't found" );
		return false;
	}

	application_pointer->device_physical			= candidate_best.device;
	application_pointer->capabilities_device		= candidate_best.capabilities;
	application_pointer->queue_family_indices		= candidate_best.queue_family_indices;
	application_pointer->image_dimension_2d_maximal	= candidate_best.image_dimension_2d_maximal;

	vkGetPhysicalDeviceMemoryProperties(
		application_pointer->device_physical, &application_pointer->memory_properties
	);

	return true;
}

static const char * vsr_queue_families_find(
		VkSurfaceKHR surface, VkPhysicalDevice device,
		struct VSR_Queue_Family_Indices * restrict out_queue_family_indices_pointer
	)
{
	assert_m( out_queue_family_indices_pointer	!= NULL, "No queue family indices storage found");

	uint32_t queue_families_amount;
	vkGetPhysicalDeviceQueueFamilyProperties( device, &queue_families_amount, NULL );

	if(	queue_families_amount == 0 )
		return "(vsr_queue_families_find) no queue families found";

	struct VkQueueFamilyProperties queue_families_array[VSR_LIMIT_STACK_FAMILIES];
	struct VkQueueFamilyProperties * queue_families_pointer;

	if(	queue_families_amount <= VSR_LIMIT_STACK_FAMILIES ) {
		queue_families_pointer = queue_families_array;
	} else {
		if(	sa_malloc_array(
				&queue_families_pointer, queue_families_amount, sizeof(*queue_families_pointer)
			) == false )
			return "(vsr_queue_families_find) allocation size overflow";
		else if ( queue_families_pointer == NULL )
			return "(vsr_queue_families_find) Queue families data allocation failed";
	}
	bool memory_was_dynamically_allocated = (queue_families_pointer != queue_families_array);

	vkGetPhysicalDeviceQueueFamilyProperties(
		device, &queue_families_amount, queue_families_pointer
	);

	struct VSR_Queue_Family_Indices indices = { 0 };
	for ( uint32_t queue_family_index = 0;
			queue_family_index < queue_families_amount; ++queue_family_index )
	{
		if(	queue_families_pointer[queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT ) {
			indices.graphics_family = queue_family_index;
			indices.has_graphics_family = true;
		}

			VkBool32 present_family_supported = false;
			if(vkGetPhysicalDeviceSurfaceSupportKHR(
					device, queue_family_index, surface, &present_family_supported
				) != VK_SUCCESS)
			{
				if(	memory_was_dynamically_allocated == true )
					free( queue_families_pointer );
				return "(vsr_queue_families_find) physical device surface support failed to get";
			}
			if(	present_family_supported == true ) {
				indices.present_family = queue_family_index;
				indices.has_present_family = true;
			}

			if((queue_families_pointer[queue_family_index].queueFlags & VK_QUEUE_TRANSFER_BIT)!= 0)
			{
				if(	(queue_families_pointer[queue_family_index].queueFlags &
						(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT )) == 0 )
				{
					indices.transfer_family = queue_family_index;
					indices.has_transfer_family = true;
				}
			}

			if(	vsr_queue_family_indices_is_complete( &indices )== true &&
				indices.has_transfer_family == true )
				break;
	}
	if(	memory_was_dynamically_allocated == true )
		free( queue_families_pointer );

	if(	indices.has_transfer_family == false && indices.has_graphics_family == true )
		indices.transfer_family = indices.graphics_family;

	*out_queue_family_indices_pointer = indices;
	return NULL;
}

static inline bool vsr_queue_family_indices_is_complete(
		struct VSR_Queue_Family_Indices * restrict queue_family_indices_pointer
	)
{
	assert_m( queue_family_indices_pointer != NULL, "No queue family indices found" );

	return
		(queue_family_indices_pointer->has_graphics_family == true &&
		queue_family_indices_pointer->has_present_family == true );
}

static const char * vsr_surface_is_support_available(
		VkSurfaceKHR surface, VkPhysicalDevice device, uint32_t * restrict formats_amount_pointer,
		uint32_t * restrict present_modes_amount_pointer
	)
{
	assert_m( formats_amount_pointer		!= NULL, "No formats amount storage found"		);
	assert_m( present_modes_amount_pointer	!= NULL, "No present modes amount storage found");

	if(	vkGetPhysicalDeviceSurfaceFormatsKHR(
			device, surface, formats_amount_pointer, NULL
		) != VK_SUCCESS )
		return "(vsr_surface_is_support_available) getting surface formats amount failed";

	return (vkGetPhysicalDeviceSurfacePresentModesKHR(
				device, surface, present_modes_amount_pointer, NULL
			) == VK_SUCCESS )
		? NULL
		: "(vsr_surface_is_support_available) getting surface present modes amount failed";
}

static bool vsr_device_extensions_get(
		VkPhysicalDevice device,
		struct VkExtensionProperties out_extensions_array_stack[static VSR_LIMIT_STACK_EXTENSIONS],
		struct VSR_Extension_Properties_Mutable * restrict out_extensions_pointer
	)
{
	assert_m( out_extensions_pointer != NULL, "No extensions storage found" );

	if(	vkEnumerateDeviceExtensionProperties(
			device, NULL, &out_extensions_pointer->amount, NULL
		) != VK_SUCCESS ||
		out_extensions_pointer->amount == 0 )
		return false;

	if(	out_extensions_pointer->amount <= VSR_LIMIT_STACK_EXTENSIONS )
		out_extensions_pointer->data_pointer = out_extensions_array_stack;
	else {
		if(	sa_malloc_array(
				&out_extensions_pointer->data_pointer, out_extensions_pointer->amount,
				sizeof(*out_extensions_pointer->data_pointer)
			) == false )
		{
			assert_mf(0, "extensions amount (%u) overflow", out_extensions_pointer->amount );
			return false;
		} else if( out_extensions_pointer->data_pointer == NULL )
			return false;
	}
	if(	vkEnumerateDeviceExtensionProperties(
			device, NULL, &out_extensions_pointer->amount, out_extensions_pointer->data_pointer
		) != VK_SUCCESS )
	{
		if(	out_extensions_pointer->data_pointer != out_extensions_array_stack )
			free( out_extensions_pointer->data_pointer );
		return false;
	}
	return true;
}

static void vsr_device_extensions_enabled_array_fill(
		struct VSR_Application				* restrict application_pointer,
		struct VSR_Extension_Names_Mutable	* restrict out_extensions_pointer
	)
{
	assert_m( application_pointer	!= NULL, "No application found"				);
	assert_m( out_extensions_pointer!= NULL, "No extensions names storage found");

	memcpy(
		out_extensions_pointer->data_pointer,
		global_device_extensions_required.data_pointer,
		global_device_extensions_required.amount * sizeof(*out_extensions_pointer->data_pointer)
	);
	out_extensions_pointer->amount = global_device_extensions_required.amount;

	if(	application_pointer->capabilities_device.has_swapchain_maintenance_1 == true ) {
		memcpy(
			out_extensions_pointer->data_pointer + out_extensions_pointer->amount,
			global_device_extensions_swapchain_maintenance_1.data_pointer,
			global_device_extensions_swapchain_maintenance_1.amount *
				sizeof(*out_extensions_pointer->data_pointer)
		);
		out_extensions_pointer->amount += VSR_EXTENSIONS_AMOUNT_DEVICE_SWAPCHAIN_MAINTENANCE_1;
	}

	if(	application_pointer->capabilities_device.has_imageless_frame_buffer == true ) {
		memcpy(
			out_extensions_pointer->data_pointer + out_extensions_pointer->amount,
			global_device_extensions_imageless_frame_buffer.data_pointer,
			global_device_extensions_imageless_frame_buffer.amount *
				sizeof(*out_extensions_pointer->data_pointer)
		);
		out_extensions_pointer->amount += VSR_EXTENSIONS_AMOUNT_DEVICE_IMAGELESS_FRAME_BUFFER;
	}
}

static void vsr_device_extension_required_check(
		VkPhysicalDevice device, PFN_vkGetPhysicalDeviceFeatures2KHR function_get_features_2,
		struct VSR_Capabilities_Device * restrict out_capabilities_pointer
	)
{
	assert_m( out_capabilities_pointer != NULL, "No capabilities storage found" );

	VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT feature_swap_chain_maintenance_1 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT
	};
	VkPhysicalDeviceImagelessFramebufferFeaturesKHR feature_imageless_frame_buffer = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES_KHR,
		.pNext = &feature_swap_chain_maintenance_1
	};
	struct VkPhysicalDeviceFeatures2 device_physical_feature_2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &feature_imageless_frame_buffer
	};

	function_get_features_2( device, &device_physical_feature_2 );

	*out_capabilities_pointer = (struct VSR_Capabilities_Device) {
		.has_swapchain_maintenance_1=
			feature_swap_chain_maintenance_1.swapchainMaintenance1 == VK_TRUE,
		.has_imageless_frame_buffer	=
			feature_imageless_frame_buffer.imagelessFramebuffer == VK_TRUE
	};
}

static bool vsr_device_present_scaling_stretch_check(
		VkPhysicalDevice device, VkSurfaceKHR surface,
		const struct VSR_Capabilities_Vulkan * restrict instance_capabilities_pointer
	)
{
	if(	instance_capabilities_pointer->get_physical_device_surface_capabilities_2 == NULL ) {
		VSR_DEBUG_LOG(
			"(vsr_device_present_scaling_stretch_check) Warning: "
			"capabilities query function is not available; assuming unsupported"
		);
		return false;
	}

	VkSurfacePresentModeKHR present_mode = {
		.sType		= VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_KHR,
		.presentMode= VK_PRESENT_MODE_FIFO_KHR
	};
	VkPhysicalDeviceSurfaceInfo2KHR surface_information = {
		.sType	= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
		.surface= surface,
		.pNext	= &present_mode
	};
	VkSurfacePresentScalingCapabilitiesEXT scaling_capabilities = {
		.sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_EXT
	};
	VkSurfaceCapabilities2KHR surface_capabilities = {
		.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR,
		.pNext = &scaling_capabilities
	};

	if(	instance_capabilities_pointer->get_physical_device_surface_capabilities_2(
			device, &surface_information, &surface_capabilities
		) != VK_SUCCESS )
	{
		VSR_DEBUG_LOG(
			"(vsr_device_present_scaling_stretch_check) Warning: "
			"querying display capabilities failed; assuming unsupported"
		);
		return false;
	}

	return
		(scaling_capabilities.supportedPresentScaling & VK_PRESENT_SCALING_STRETCH_BIT_EXT) != 0;
}

static const char * vsr_device_suitability_rate(
		struct VSR_Application * restrict application_pointer, VkPhysicalDevice device,
		struct VSR_Device_Candidate * restrict out_candidate_pointer
	)
{
	assert_m(application_pointer	!= NULL, "No application found"		);
	assert_m(out_candidate_pointer	!= NULL, "No scores storage found"	);

	VkPhysicalDeviceProperties device_properties;
	vkGetPhysicalDeviceProperties( device, &device_properties );
	if(	device_properties.limits.maxImageDimension2D > VSR_LIMIT_EXTENT_MAXIMAL )
		return "(vsr_device_suitability_rate) maxImageDimension2D exceeds hard limit";

	const char * error_message_pointer;
	if((error_message_pointer = vsr_queue_families_find(
			application_pointer->surface, device, &out_candidate_pointer->queue_family_indices
		)) != NULL )
		return error_message_pointer;

	out_candidate_pointer->device = device;
	out_candidate_pointer->score = 0;
	if(	vsr_queue_family_indices_is_complete(
			&out_candidate_pointer->queue_family_indices
		) == false ||
		vsr_device_capabilities_build(
			device, application_pointer->surface,
			&application_pointer->capabilities_vulkan, &out_candidate_pointer->capabilities
		) == false )
		return NULL;

	uint32_t formats_amount, present_modes_amount;
	if((error_message_pointer = vsr_surface_is_support_available(
			application_pointer->surface, device, &formats_amount, &present_modes_amount
		)) != NULL )
		return error_message_pointer;
	if(	formats_amount == 0 || present_modes_amount == 0 )
		return NULL;

	if(	device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU )
		out_candidate_pointer->score += 1000;
	if(	out_candidate_pointer->queue_family_indices.has_transfer_family== true )
		out_candidate_pointer->score += 500;
	if(	out_candidate_pointer->capabilities.has_imageless_frame_buffer == true )
		out_candidate_pointer->score += 250;
	if(	out_candidate_pointer->capabilities.has_swapchain_maintenance_1== true )
		out_candidate_pointer->score += 250;
	if(	out_candidate_pointer->capabilities.has_present_scaling_stretch== true )
		out_candidate_pointer->score += 125;

	out_candidate_pointer->score += device_properties.limits.maxImageDimension2D;
	out_candidate_pointer->image_dimension_2d_maximal =
		device_properties.limits.maxImageDimension2D;

#ifndef NDEBUG
	vsr_debug_gpu_print(
		device_properties, out_candidate_pointer->capabilities, out_candidate_pointer->score
	);
#endif

	return NULL;
}

static void vsr_swap_chain_support_details_free(
		struct VSR_Swap_Chain_Support_Details * restrict swap_chain_support_pointer
	)
{
	assert_m( swap_chain_support_pointer != NULL, "No swap chain support data found" );

	if(	swap_chain_support_pointer->surface_formats_pointer ) {
		free( swap_chain_support_pointer->surface_formats_pointer );
		swap_chain_support_pointer->surface_formats_pointer = NULL;
	}
	if(	swap_chain_support_pointer->present_modes_pointer ) {
		free( swap_chain_support_pointer->present_modes_pointer );
		swap_chain_support_pointer->present_modes_pointer = NULL;
	}
}

static const char * vsr_swap_chain_support_query(
		VkSurfaceKHR surface, VkPhysicalDevice device,
		struct VSR_Swap_Chain_Support_Details * restrict out_swap_chain_support_details_pointer
	)
{
	assert_m(out_swap_chain_support_details_pointer != NULL,"No swapchain support storage found");

	if(	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
			device, surface, &out_swap_chain_support_details_pointer->surface_capabilities
		) != VK_SUCCESS )
		return "(vsr_swap_chain_support_query) surface capabilities failed to get";

	uint32_t formats_amount;
	if(	vkGetPhysicalDeviceSurfaceFormatsKHR(
			device, surface, &formats_amount, NULL
		) != VK_SUCCESS )
		return "(vsr_swap_chain_support_query) surface formats failed to get amount";

	if(	formats_amount != 0 ) {
		if(	sa_malloc_array(
				&out_swap_chain_support_details_pointer->surface_formats_pointer,
				formats_amount,
				sizeof(*out_swap_chain_support_details_pointer->surface_formats_pointer)
			) == false )
			return "(vsr_swap_chain_support_query) allocation size overflow";
		else if ( out_swap_chain_support_details_pointer->surface_formats_pointer == NULL )
			return "(vsr_swap_chain_support_query) surface format allocation failed";

		if(	vkGetPhysicalDeviceSurfaceFormatsKHR(
				device, surface, &formats_amount,
				out_swap_chain_support_details_pointer->surface_formats_pointer
			) != VK_SUCCESS )
		{
			vsr_swap_chain_support_details_free( out_swap_chain_support_details_pointer );
			return "(vsr_swap_chain_support_query) surface formats failed to get";
		}
	}
	out_swap_chain_support_details_pointer->formats_amount = formats_amount;

	uint32_t present_modes_amount;
	if(	vkGetPhysicalDeviceSurfacePresentModesKHR(
			device, surface, &present_modes_amount, NULL
		) != VK_SUCCESS )
	{
		vsr_swap_chain_support_details_free( out_swap_chain_support_details_pointer );
		return "(vsr_swap_chain_support_query) surface present modes failed to get amount";
	}

	if(	present_modes_amount != 0 ) {
		if(	sa_malloc_array(
				&out_swap_chain_support_details_pointer->present_modes_pointer,
				present_modes_amount,
				sizeof(*out_swap_chain_support_details_pointer->present_modes_pointer)
			) == false )
		{
			vsr_swap_chain_support_details_free( out_swap_chain_support_details_pointer );
			return "(vsr_swap_chain_support_query) allocation size overflow";
		} else if ( out_swap_chain_support_details_pointer->present_modes_pointer == NULL ) {
			vsr_swap_chain_support_details_free( out_swap_chain_support_details_pointer );
			return "(vsr_swap_chain_support_query) present modes allocation failed";
		}

		if(	vkGetPhysicalDeviceSurfacePresentModesKHR(
				device, surface, &present_modes_amount,
				out_swap_chain_support_details_pointer->present_modes_pointer
			) != VK_SUCCESS)
		{
			vsr_swap_chain_support_details_free( out_swap_chain_support_details_pointer );
			return "(vsr_swap_chain_support_query) getting surface present modes failed";
		}
	}
	out_swap_chain_support_details_pointer->present_modes_amount = present_modes_amount;

	return NULL;
}

static const char * vsr_synchronization_frames_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	if(	sa_malloc_array(
			&application_pointer->synchronization_frame_pointer,
			application_pointer->frames_in_flight_limit,
			sizeof(*application_pointer->synchronization_frame_pointer)
		) == false )
		return "(vsr_synchronization_frames_create) allocation size overflow";
	else if ( application_pointer->synchronization_frame_pointer == NULL )
		return "(vsr_synchronization_frames_create) synchronization objects allocation failed";

	for(uint8_t frame_index = 0;
			frame_index < application_pointer->frames_in_flight_limit;
				++frame_index )
	{
		struct VSR_Synchronization_Frame * synchronization_frame_pointer =
			&application_pointer->synchronization_frame_pointer[frame_index];
		const char * error_message_pointer;
		if((error_message_pointer = vsr_synchronization_frame_create(
				application_pointer, synchronization_frame_pointer
			)) != NULL )
		{
			vsr_synchronization_frames_destroy(application_pointer, frame_index);
			return error_message_pointer;
		}
	}

	return NULL;
}

static const char * vsr_synchronization_fence_present_create(
		struct VSR_Application * restrict application_pointer,
		VkFence ** restrict out_fences_pointer, uint32_t fences_to_create_amount
	)
{
	assert_m( application_pointer!= NULL, "No application found"	);
	assert_m( out_fences_pointer != NULL, "No fences storage found"	);

	if(	sa_malloc_array(
			out_fences_pointer, fences_to_create_amount, sizeof(**out_fences_pointer)
		) == false )
		return "(vsr_synchronization_fence_present_create) allocation size overflow";
	else if ( *out_fences_pointer == NULL )
		return
			"(vsr_synchronization_fence_present_create) fences memory allocation failed";

	struct VkFenceCreateInfo fence_create_information = {
		.sType	= VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags	= VK_FENCE_CREATE_SIGNALED_BIT
	};

	for(uint32_t fence_index_create = 0;
			fence_index_create < fences_to_create_amount;
				++fence_index_create )
	{
		if(	vkCreateFence(
				application_pointer->device, &fence_create_information, NULL,
				&(*out_fences_pointer)[fence_index_create]
			) != VK_SUCCESS )
		{
			vsr_synchronization_fence_present_destroy(
				application_pointer->device, *out_fences_pointer, fence_index_create
			);
			*out_fences_pointer = NULL;
			return "(vsr_synchronization_fence_present_create) fences creation failed";
		}
	}

	return NULL;
}

static void vsr_synchronization_fence_present_destroy(
		VkDevice device, VkFence * restrict fences_pointer, uint32_t fences_amount
	)
{
	for(uint32_t fence_index_destroy = 0;
			fence_index_destroy < fences_amount;
				++fence_index_destroy )
		vkDestroyFence( device, fences_pointer[fence_index_destroy], NULL );

	free( fences_pointer );
}

static const char * vsr_synchronization_semaphores_render_finished_create(
		struct VSR_Application * restrict application_pointer,
		VkSemaphore ** restrict out_semaphores_pointer, uint32_t semaphores_amount
	)
{
	assert_m( application_pointer	!= NULL, "No application found"			);
	assert_m( out_semaphores_pointer!= NULL, "No semaphores storage found"	);

	if(	sa_malloc_array(
			out_semaphores_pointer, semaphores_amount, sizeof(**out_semaphores_pointer)
		) == false )
		return "(vsr_synchronization_semaphores_render_finished_create) allocation size overflow";
	else if ( *out_semaphores_pointer == NULL )
		return
			"(vsr_synchronization_semaphores_render_finished_create) "
			"semaphores memory allocation failed";

	struct VkSemaphoreCreateInfo semaphore_create_information = {
		.sType	= VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};
	for(uint32_t semaphore_index_create = 0;
			semaphore_index_create < semaphores_amount;
				++semaphore_index_create )
	{
		if(	vkCreateSemaphore(
				application_pointer->device, &semaphore_create_information, NULL,
				&(*out_semaphores_pointer)[semaphore_index_create]
			) != VK_SUCCESS )
		{
			vsr_synchronization_semaphores_render_finished_destroy(
				application_pointer->device, *out_semaphores_pointer, semaphore_index_create
			);
			return
				"(vsr_synchronization_semaphores_render_finished_create) "
				"semaphores creation failed";
		}
	}

	return NULL;
}

static void vsr_synchronization_semaphores_render_finished_destroy(
		VkDevice device, VkSemaphore * restrict semaphores_pointer, uint32_t semaphores_amount
	)
{
	assert_m( semaphores_pointer != NULL, "No semaphores found"	);

	for(uint32_t semaphore_index_destroy = 0;
			semaphore_index_destroy < semaphores_amount;
				++semaphore_index_destroy )
		vkDestroySemaphore( device, semaphores_pointer[semaphore_index_destroy], NULL );

	free( semaphores_pointer );
}

static const char * vsr_synchronization_frame_create(
		const struct VSR_Application * restrict application_pointer,
		struct VSR_Synchronization_Frame * restrict synchronization_frame_pointer
	)
{
	assert_m( application_pointer			!= NULL, "No application found"						);
	assert_m( synchronization_frame_pointer	!= NULL, "No synchronization object storage found"	);

	struct VkSemaphoreCreateInfo semaphore_create_information = {
		.sType	= VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};

	struct VkFenceCreateInfo fence_create_information = {
		.sType	= VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags	= VK_FENCE_CREATE_SIGNALED_BIT
	};

	if(	vkCreateSemaphore(
			application_pointer->device, &semaphore_create_information, NULL,
			&synchronization_frame_pointer->image_available_semaphore
		) != VK_SUCCESS )
		return "(vsr_synchronization_frame_create) image available semaphore creation failed";

	if(	vkCreateFence(
			application_pointer->device, &fence_create_information, NULL,
			&synchronization_frame_pointer->in_flight_fence
		) != VK_SUCCESS )
	{
		vkDestroySemaphore(
			application_pointer->device,
			synchronization_frame_pointer->image_available_semaphore, NULL
		);
		return "(vsr_synchronization_frame_create) in flight fence creation failed";
	}

	return NULL;
}

static void vsr_synchronization_frames_destroy(
		struct VSR_Application * restrict application_pointer, uint8_t frames_amount
	)
{
	assert_m( application_pointer != NULL, "No application found" );
	assert_m(
		frames_amount <= application_pointer->frames_in_flight_limit,
		"Excess frames to delete"
	);

	for ( uint8_t frame_index = 0; frame_index < frames_amount; ++frame_index ) {
		struct VSR_Synchronization_Frame * synchronization_frame_pointer =
			&application_pointer->synchronization_frame_pointer[frame_index];
		vkDestroySemaphore(
			application_pointer->device,
			synchronization_frame_pointer->image_available_semaphore, NULL
		);
		vkDestroyFence(
			application_pointer->device, synchronization_frame_pointer->in_flight_fence, NULL
		);
	}

	free( application_pointer->synchronization_frame_pointer );
	application_pointer->synchronization_frame_pointer = NULL;
}

static void vsr_projection_refresh(struct VSR_Application * restrict application_pointer) {
	assert_m( application_pointer != NULL, "No application found" );

	float
		width = (application_pointer->frame_state.width > 0)
			? (float) application_pointer->frame_state.width
			: 1.f,
		height = (application_pointer->frame_state.height > 0)
			? (float) application_pointer->frame_state.height
			: 1.f;
	float aspect = width / (float) height;
	float minimal_side = fminf( 1.f, aspect );
	float vertical_field_of_view = 2.f * (atanf(tanf(glm_rad(45.f) * 0.5f) / minimal_side));
	glm_perspective(
		vertical_field_of_view, aspect, 0.1f, 10.f, application_pointer->cached_projection
	);
	application_pointer->cached_projection[1][1] *= -1.f;
	glm_mat4_mul(
		application_pointer->cached_projection, application_pointer->cached_view,
		application_pointer->cached_projection_view
	);
	application_pointer->frame_state.is_projection_dirty = false;
}

static bool vsr_buffer_uniform_update(
		struct VSR_Application * restrict application_pointer, uint32_t current_frame
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	application_pointer->spin_angle_current = fmodf(
		application_pointer->spin_angle_current + application_pointer->spin_angle_rotation,
		VSR_LIMIT_TURNOVER
	);

	struct VSR_Uniform_Buffer_Object buffer_uniform_object;
	glm_mat4_copy(
		application_pointer->cached_projection_view, buffer_uniform_object.view_projection
	);
	glm_rotate_make(
		buffer_uniform_object.model, application_pointer->spin_angle_current,
		(vec3){ 0.5f, 1.f, 0.5f }
	);

	/* in current context overflow is almost impossible */
	VkDeviceSize size_offset =
		current_frame * application_pointer->buffer_uniform_size_alignment;

	uint8_t * mapped_memory_pointer = (uint8_t *)
		application_pointer->buffers_uniform_mapped_pointer;
	memcpy(
		mapped_memory_pointer + size_offset,
		&buffer_uniform_object,
		application_pointer->buffer_uniform_size
	);

	if(	application_pointer->is_buffer_uniform_coherent == false ) {
		struct VkMappedMemoryRange flush_range = {
			.sType	= VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
			.memory	= application_pointer->buffers_uniform_memory,
			.offset	= size_offset,
			.size	= application_pointer->buffer_uniform_size_flush
		};
		if(	vkFlushMappedMemoryRanges(
				application_pointer->device, 1, &flush_range
			) != VK_SUCCESS )
			return false;
	}
	return true;
}

static void vsr_frame_discard(
		struct VSR_Application * restrict application_pointer, uint32_t image_index
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	VkSemaphore wait_semaphores_pointer[] = {
		application_pointer->synchronization_frame_pointer[application_pointer->current_frame].
			image_available_semaphore
	};
	struct VkPresentInfoKHR presentation_information = {
		.sType				= VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount	= 1,
		.pWaitSemaphores	= wait_semaphores_pointer,
		.swapchainCount		= 1,
		.pSwapchains		= &(VkSwapchainKHR){ application_pointer->swap_chain_data.swap_chain },
		.pImageIndices		= &image_index
	};
	VkResult result = vkQueuePresentKHR(
		application_pointer->present_queue, &presentation_information
	);
	if(	result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR )
		vsr_frame_render_failed(
			application_pointer, "(vsr_frame_discard) discarding frame failed"
		);
	else ++application_pointer->frame_discarded_amount;
}

static VkResult vsr_frame_fences_wait(
		const struct VSR_Application * restrict application_pointer,
		const struct VSR_Synchronization_Frame * restrict synchronization_frame_pointer
	)
{
	assert_m( application_pointer			!= NULL, "No application found"						);
	assert_m( synchronization_frame_pointer!= NULL, "No synchronization object storage found"	);

	if(	application_pointer->capabilities_device.has_swapchain_maintenance_1 == true )
		return vkWaitForFences(
			application_pointer->device, 2,
			(VkFence[]) {
				synchronization_frame_pointer->in_flight_fence,
				application_pointer->present_fences_pointer[application_pointer->current_frame]
			},
			VK_TRUE, VSR_LIMIT_TIME_WAIT_FENCE
		);
	return vkWaitForFences(
		application_pointer->device, 1,
		&synchronization_frame_pointer->in_flight_fence,
		VK_TRUE, VSR_LIMIT_TIME_WAIT_FENCE
	);
}

static bool vsr_frame_fence_present_reset(struct VSR_Application * restrict application_pointer) {
	assert_m( application_pointer != NULL, "No application found" );

	if(	application_pointer->capabilities_device.has_swapchain_maintenance_1 == false )
		return true;

	if(	vkResetFences(
			application_pointer->device, 1,
			&application_pointer->present_fences_pointer[application_pointer->current_frame]
		) == VK_SUCCESS )
		return true;

	if(	vsr_device_recreate(application_pointer) == false )
		vsr_frame_render_failed(
			application_pointer, "(vsr_frame_fence_present_reset) present fence reset failed"
		);
	return false;
}

static void vsr_frame_draw(struct VSR_Application * restrict application_pointer) {
	assert_m( application_pointer != NULL, "No application found" );

	if(	application_pointer->frame_discarded_amount >= VSR_LIMIT_FAILURES_FRAME_DISCARD ) {
		if(	vsr_device_recreate( application_pointer ) == false )
			vsr_frame_render_failed(
				application_pointer, "(vsr_frame_draw) drawing fails limit exceeded"
			);
		return;
	}

	/* only works properly because of scaling */
	if(	atomic_load_explicit(
			&application_pointer->is_swap_chain_valid, memory_order_acquire
		) == false &&
		vsr_swap_chain_recreate( application_pointer ) == false )
		return;

	const struct VSR_Synchronization_Frame * synchronization_frame_pointer =
		&application_pointer->synchronization_frame_pointer[application_pointer->current_frame];

	if(	vsr_frame_fences_wait(
			application_pointer, synchronization_frame_pointer
		) != VK_SUCCESS )
	{
		if(	vsr_device_recreate(application_pointer) == false )
			vsr_frame_render_failed(
				application_pointer, "(vsr_frame_draw) waiting for in-flight fences failed"
			);
		return;
	}

	uint32_t image_index;
	if(	vsr_frame_image_acquire(
			application_pointer, synchronization_frame_pointer, &image_index
		) == false )
		return;

	if(	vsr_buffer_uniform_update(
			application_pointer, application_pointer->current_frame
		) == false )
	{
		VSR_DEBUG_LOG("(vsr_frame_draw) uniform buffer flush failed; skipping frame");
		vsr_frame_discard( application_pointer, image_index );
		return;
	}

	/* make sure command buffer is able to be recorded */
	if(	vkResetCommandBuffer(
			application_pointer->command_buffers_pointer[application_pointer->current_frame], 0
		) != VK_SUCCESS )
	{
		if(	vsr_device_recreate(application_pointer) == false )
			vsr_frame_render_failed(
				application_pointer, "(vsr_frame_draw) command buffer reset failed"
			);
		return;
	}

	if(	vsr_command_buffer_record(
			application_pointer,
			application_pointer->command_buffers_pointer[application_pointer->current_frame],
			image_index
		) == false )
	{
		vsr_frame_discard( application_pointer, image_index );
		return;
	}

	if(	vsr_frame_commands_submit(
			application_pointer, synchronization_frame_pointer, image_index
		) == false ||
		vsr_frame_image_present(application_pointer, image_index) == false )
		return;

	vsr_delay_deletion_process( application_pointer );

	application_pointer->frame_discarded_amount = 0;

	if(	++application_pointer->current_frame >= application_pointer->frames_in_flight_limit )
		application_pointer->current_frame = 0;
}

static VkSemaphore vsr_frame_get_render_finished_semaphore(
		const struct VSR_Application * restrict application_pointer,
		uint32_t frame_index, uint32_t image_index
	)
{
	return application_pointer->render_finished_semaphores_pointer[
		(application_pointer->capabilities_device.has_swapchain_maintenance_1 == true)
		? frame_index
		: image_index
	];
}

static void vsr_frame_render_failed(
		struct VSR_Application * restrict application_pointer,
		const char * restrict error_message_pointer
	)
{
	assert_m( application_pointer	!= NULL, "No application found"		);
	assert_m( error_message_pointer	!= NULL, "No error message found"	);

	woem_push( "%s", error_message_pointer );

	pthread_mutex_lock( &application_pointer->render_mutex );
	application_pointer->is_running = false;
	pthread_mutex_unlock( &application_pointer->render_mutex );

	atomic_store_explicit(
		&application_pointer->is_render_failed, true, memory_order_relaxed
	);
	glfwPostEmptyEvent();
}

static bool vsr_frame_image_present(
		struct VSR_Application * restrict application_pointer, uint32_t image_index
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	if(	vsr_frame_fence_present_reset(application_pointer) == false )
		return false;

	VkSwapchainPresentFenceInfoEXT presentation_fence_information = {
		.sType			= VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
		.swapchainCount	= 1,
		.pFences		=
			&application_pointer->present_fences_pointer[application_pointer->current_frame]
	};

	VkSemaphore signal_semaphores_array[] = {
		vsr_frame_get_render_finished_semaphore(
			application_pointer, application_pointer->current_frame, image_index
		)
	};

	VkSwapchainKHR swap_chains_array[] = { application_pointer->swap_chain_data.swap_chain };
	struct VkPresentInfoKHR presentation_information = {
		.sType				= VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext				=
			(application_pointer->capabilities_device.has_swapchain_maintenance_1 == true)
				? &presentation_fence_information
				: NULL,
		.waitSemaphoreCount	= 1,
		.pWaitSemaphores	= signal_semaphores_array,
		.swapchainCount		= 1,
		.pSwapchains		= swap_chains_array,
		.pImageIndices		= &image_index
	};

	VkResult result = vkQueuePresentKHR(
		application_pointer->present_queue, &presentation_information
	);
	return vsr_frame_image_present_result_handle( application_pointer, result );
}

static bool vsr_frame_image_present_result_handle(
		struct VSR_Application * restrict application_pointer, VkResult result
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	switch( result ) {
	case VK_ERROR_SURFACE_LOST_KHR: {
		if(	vsr_surface_recreate( application_pointer ) == false ) {
			vsr_frame_render_failed(
				application_pointer, "(vsr_frame_image_present) surface recreation failed"
			);
			return false;
		}
	/* fall through */
	case VK_ERROR_OUT_OF_DATE_KHR:
		if(	vsr_synchronization_fence_present_recreate(application_pointer) == false )
			return false;
		atomic_store_explicit(
			&application_pointer->is_swap_chain_valid, false, memory_order_relaxed
		);
	/* fall through */
	case VK_SUCCESS: {
		return true;
	}
	case VK_SUBOPTIMAL_KHR:
		if(	application_pointer->capabilities_device.has_present_scaling_stretch == false )
			atomic_store_explicit(
				&application_pointer->is_swap_chain_valid, false, memory_order_relaxed
			);
		return true;
	}
	case VK_ERROR_DEVICE_LOST: {
		if(	vsr_device_recreate( application_pointer ) == false )
			vsr_frame_render_failed(
				application_pointer, "(vsr_frame_image_present) the GPU device has been lost"
			);
		return false;
	}
	default: {
		vsr_frame_render_failed(
			application_pointer, "(vsr_frame_image_present) presentation failed"
		);
		return false;
	}
	}
}

static bool vsr_synchronization_fence_present_recreate(
		struct VSR_Application * restrict application_pointer
	)
{
	if(	application_pointer->capabilities_device.has_swapchain_maintenance_1 == false )
		return true;

	if(	application_pointer->present_fences_pointer != NULL ) {
		if(	vkQueueWaitIdle(application_pointer->present_queue) != VK_SUCCESS ) {
			vsr_frame_render_failed(
				application_pointer,
				"(vsr_synchronization_fence_present_recreate) present queue wait failed"
			);
			return false;
		}
		vsr_synchronization_fence_present_destroy(
			application_pointer->device, application_pointer->present_fences_pointer,
			application_pointer->frames_in_flight_limit
		);
		application_pointer->present_fences_pointer = NULL;
	}

	const char * error_message_pointer;
	if((error_message_pointer = vsr_synchronization_fence_present_create(
			application_pointer, &application_pointer->present_fences_pointer,
			application_pointer->frames_in_flight_limit
		)) != NULL )
	{
		vsr_frame_render_failed(application_pointer, error_message_pointer);
		return false;
	}

	return true;
}

static bool vsr_frame_image_acquire(
		struct VSR_Application * restrict application_pointer,
		const struct VSR_Synchronization_Frame * restrict synchronization_frame_pointer,
		uint32_t * restrict out_image_index
	)
{
	assert_m( out_image_index				!= NULL, "No image index found"						);
	assert_m( application_pointer			!= NULL, "No application found"						);
	assert_m( synchronization_frame_pointer!= NULL, "No synchronization object storage found"	);

	for( uint32_t attempt = 0; attempt < VSR_LIMIT_IMAGE_ACQUIRE_ATTEMPTS; ++attempt ) {
		pthread_mutex_lock( &application_pointer->render_mutex );

		bool is_minimized = application_pointer->is_minimized;

		pthread_mutex_unlock( &application_pointer->render_mutex );

		if(	is_minimized == true )
			return false;

		VkResult result = vkAcquireNextImageKHR(
			application_pointer->device, application_pointer->swap_chain_data.swap_chain,
			VSR_LIMIT_TIME_WAIT_ACQUIRE, synchronization_frame_pointer->image_available_semaphore,
			VK_NULL_HANDLE, out_image_index
		);
		switch( result ) {
		case VK_SUCCESS: {
		case VK_SUBOPTIMAL_KHR:
			return true;
		}
		case VK_ERROR_OUT_OF_DATE_KHR: {
			if(	vsr_swap_chain_recreate( application_pointer ) == false )
				return false;
			break;
		}
		case VK_ERROR_SURFACE_LOST_KHR: {
			if(	vsr_surface_recreate( application_pointer ) == false ) {
				vsr_frame_render_failed(
					application_pointer, "(vsr_frame_image_acquire) surface recreation failed"
				);
				return false;
			}
			if(	vsr_swap_chain_recreate( application_pointer ) == false )
				return false;
			break;
		}
		case VK_TIMEOUT: {
			VSR_DEBUG_LOG("(vsr_frame_image_acquire) acquiring frame image timed out");
			break;
		}
		case VK_ERROR_DEVICE_LOST: {
			if(	vsr_device_recreate( application_pointer ) == false )
				vsr_frame_render_failed(
					application_pointer, "(vsr_frame_image_acquire) the GPU device has been lost"
				);
			return false;
		}
		default: {
			vsr_frame_render_failed(
				application_pointer, "(vsr_frame_image_acquire) acquiring swap chain image failed"
			);
			return false;
		}
		}
	}

	VSR_DEBUG_LOG( "(vsr_frame_image_acquire) acquiring frame image failed" );

	if(	vsr_device_recreate(application_pointer) == false )
		vsr_frame_render_failed(
			application_pointer,
			"(vsr_frame_image_acquire) acquiring frame image exceed the limit"
		);
	return false;
}

static bool vsr_surface_recreate(struct VSR_Application * restrict application_pointer) {
	assert_m( application_pointer != NULL, "No application found" );

	vsr_swap_chain_data_associated_destroy(application_pointer);

	vkDestroySurfaceKHR(
		application_pointer->instance, application_pointer->surface, NULL
	);
	application_pointer->surface = NULL;

	return vsr_surface_create( application_pointer );
}

static bool vsr_frame_commands_submit(
		struct VSR_Application * restrict application_pointer,
		const struct VSR_Synchronization_Frame * restrict synchronization_frame_pointer,
		uint32_t image_index
	)
{
	assert_m( application_pointer			!= NULL, "No application found"						);
	assert_m( synchronization_frame_pointer	!= NULL, "No synchronization object storage found"	);

	if(	vkResetFences(
			application_pointer->device, 1, &synchronization_frame_pointer->in_flight_fence
		) != VK_SUCCESS )
	{
		if(	vsr_device_recreate(application_pointer) == false )
			vsr_frame_render_failed(
				application_pointer, "(vsr_frame_commands_submit) in-flight fence reset failed"
			);
		return false;
	}

	VkSemaphore wait_semaphores_array[] = {
		synchronization_frame_pointer->image_available_semaphore
	};
	VkPipelineStageFlags wait_stages_array[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

	VkSemaphore signal_semaphores_array[] = {
		vsr_frame_get_render_finished_semaphore(
			application_pointer, application_pointer->current_frame, image_index
		)
	};

	struct VkSubmitInfo submit_information = {
		.sType					= VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount		= 1,
		.pWaitSemaphores		= wait_semaphores_array,
		.pWaitDstStageMask		= wait_stages_array,
		.commandBufferCount		= 1,
		.pCommandBuffers =
			&application_pointer->command_buffers_pointer[application_pointer->current_frame],
		.signalSemaphoreCount	= 1,
		.pSignalSemaphores		= signal_semaphores_array
	};

	if(	vkQueueSubmit(
			application_pointer->graphics_queue, 1, &submit_information,
			synchronization_frame_pointer->in_flight_fence
		) != VK_SUCCESS )
	{
		if(	vkQueueSubmit(
				application_pointer->graphics_queue, 0, NULL,
				synchronization_frame_pointer->in_flight_fence
			) == VK_SUCCESS )
		{
			vsr_frame_discard( application_pointer, image_index );
			return false;
		}
		if(	vsr_device_recreate(application_pointer) == false )
			vsr_frame_render_failed(
				application_pointer,
				"(vsr_frame_commands_submit) draw command buffer submission failed"
			);
		return false;
	}
	return true;
}

static bool vsr_command_buffer_record(
		struct VSR_Application * restrict application_pointer, VkCommandBuffer command_buffer,
		uint32_t image_index
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	struct VkCommandBufferBeginInfo command_buffer_begin_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	};

	if(	vkBeginCommandBuffer( command_buffer, &command_buffer_begin_information ) != VK_SUCCESS)
	{
		VSR_DEBUG_LOG("(vsr_command_buffer_record) recording command buffer beginning failed");
		return false;
	}

	VkClearValue clear_color = {
		.color = {{ 1.0f, 0.0f, 0.0f, 1.0f }}
	};

	VkRenderPassAttachmentBeginInfoKHR attachment_begin_information = {
		.sType				= VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO_KHR,
		.attachmentCount	= VSR_ATTACHMENT_COLOR_AMOUNT,
		.pAttachments		= &application_pointer->swap_chain_image_views_pointer[image_index]
	};

	struct VkRenderPassBeginInfo render_pass_information = {
		.sType				= VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass			= application_pointer->render_pass,
		.renderArea 		= {
			.extent	= application_pointer->swap_chain_data.extent
		},
		.clearValueCount	= 1,/* values to VK_ATTACHMENT_LOAD_OP_CLEAR */
		.pClearValues		= &clear_color
	};

	if(	application_pointer->capabilities_device.has_imageless_frame_buffer == true ) {
		render_pass_information.pNext		= &attachment_begin_information;
		render_pass_information.framebuffer = application_pointer->swap_chain_data.frame_buffer;
	} else
		render_pass_information.framebuffer =
			application_pointer->swap_chain_data.frame_buffers_pointer[image_index];

	vkCmdBeginRenderPass(command_buffer, &render_pass_information, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(
		command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, application_pointer->graphics_pipeline
	);

	struct VkViewport viewport = {
		.width		= (float) application_pointer->swap_chain_data.extent.width,
		.height		= (float) application_pointer->swap_chain_data.extent.height,
		.maxDepth	= 1.0f
	};
	vkCmdSetViewport( command_buffer, 0, 1, &viewport );

	struct VkRect2D scissor = {
		.extent	= application_pointer->swap_chain_data.extent
	};
	vkCmdSetScissor( command_buffer, 0, 1, &scissor );

	vkCmdBindVertexBuffers(
		command_buffer, 0, 1, &(VkBuffer){application_pointer->buffer_vertex}, &(VkDeviceSize){0}
	);
	vkCmdBindIndexBuffer(
		command_buffer, application_pointer->buffer_index, 0, VK_INDEX_TYPE_UINT16
	);

	vkCmdBindDescriptorSets(
		command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, application_pointer->pipeline_layout, 0,
		1, &application_pointer->descriptor_sets_pointer[application_pointer->current_frame], 0,
		NULL
	);

	vkCmdDrawIndexed(
		command_buffer,
		(uint32_t)(sizeof(global_indices_array) / sizeof(global_indices_array[0])), 1, 0, 0, 0
	);

	vkCmdEndRenderPass( command_buffer );

	if(	vkEndCommandBuffer( command_buffer ) != VK_SUCCESS ) {
		VSR_DEBUG_LOG( "(vsr_command_buffer_record) command buffer recording failed" );
		return false;
	}

	return true;
}

static const char * vsr_command_buffers_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	if(	sa_malloc_array(
			&application_pointer->command_buffers_pointer,
			application_pointer->frames_in_flight_limit,
			sizeof(*application_pointer->command_buffers_pointer)
		) == false )
		return "(vsr_command_buffers_create) allocation size overflow";
	else if ( application_pointer->command_buffers_pointer == NULL )
		return "(vsr_command_buffers_create) command buffer memory allocation failed";

	struct VkCommandBufferAllocateInfo allocation_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool		= application_pointer->command_pool_graphic,
		.level				= VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount	= (uint32_t) application_pointer->frames_in_flight_limit
	};

	return( vkAllocateCommandBuffers(
				application_pointer->device, &allocation_information,
				application_pointer->command_buffers_pointer
			) != VK_SUCCESS )
		? "(vsr_command_buffers_create) command buffer allocation failed"
		: NULL;
}

static const char * vsr_inclusive_command_pool_create(
		struct VSR_Application * restrict application_pointer,
		VkCommandPool * restrict command_pool_pointer, VkCommandPoolCreateFlags flags,
		uint32_t family
	)
{
	assert_m( application_pointer	!= NULL, "No application found"				);
	assert_m( command_pool_pointer	!= NULL, "No command pool found"			);

	struct VkCommandPoolCreateInfo pool_create_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags				= flags,
		.queueFamilyIndex	= family
	};

	if(	vkCreateCommandPool_wrapped(
			application_pointer->device, &pool_create_information, NULL, command_pool_pointer
		) != VK_SUCCESS)
		return "(vsr_inclusive_command_pool_create) command pool creation failed";

	return NULL;
}

static const char * vsr_command_pools_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	const char * error_message_pointer;

	if((error_message_pointer = vsr_inclusive_command_pool_create(
			application_pointer, &application_pointer->command_pool_graphic,
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			application_pointer->queue_family_indices.graphics_family
		)) != NULL )
		return error_message_pointer;

	return
		(application_pointer->queue_family_indices.has_transfer_family == true)
		? vsr_inclusive_command_pool_create(
			application_pointer, &application_pointer->command_pool_transfer,
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			application_pointer->queue_family_indices.transfer_family
		)
		: NULL;
}

static const char * vsr_frame_buffer_create_imaged(
		struct VSR_Application * restrict application_pointer,
		VkImageView * restrict swap_chain_image_views_pointer,
		struct VSR_Swap_Chain_Data * restrict out_swap_chain_data_pointer,
		struct VkFramebufferCreateInfo frame_buffer_create_information
	)
{
	assert_m( application_pointer			!= NULL, "No application found"			);
	assert_m( out_swap_chain_data_pointer	!= NULL, "No swap chain data found"		);
	assert_m( swap_chain_image_views_pointer!= NULL, "No image views storage found"	);

	uint32_t frames_amount = out_swap_chain_data_pointer->image_views_amount;
	VkFramebuffer ** frame_buffers_pointer = &out_swap_chain_data_pointer->frame_buffers_pointer;

	if(	sa_malloc_array(
			frame_buffers_pointer, frames_amount, sizeof(**frame_buffers_pointer)
		) == false )
		return "(vsr_frame_buffer_create_imaged) allocation size overflow";
	else if ( *frame_buffers_pointer == NULL )
		return "(vsr_frame_buffer_create_imaged) frame buffers allocation failed";

	for(uint32_t frame_buffer_index = 0; frame_buffer_index < frames_amount; ++frame_buffer_index)
	{
		frame_buffer_create_information.pAttachments =
			&swap_chain_image_views_pointer[frame_buffer_index];

		if(	vkCreateFramebuffer(
				application_pointer->device, &frame_buffer_create_information,
				NULL, &(*frame_buffers_pointer)[frame_buffer_index]
			) != VK_SUCCESS )
		{
			for( uint32_t cleanup_index = 0; cleanup_index < frame_buffer_index; ++cleanup_index )
				vkDestroyFramebuffer(
					application_pointer->device, (*frame_buffers_pointer)[cleanup_index], NULL
				);
			free( *frame_buffers_pointer );
			*frame_buffers_pointer = NULL;

			return "(vsr_frame_buffer_create_imaged) frame buffers creation failed";
		}
	}

	return NULL;
}

static const char * vsr_frame_buffer_create(
		struct VSR_Application * restrict application_pointer,
		VkImageView * restrict swap_chain_image_views_pointer,
		struct VSR_Swap_Chain_Data * restrict out_swap_chain_data_pointer
	)
{
	assert_m( application_pointer			!= NULL, "No application found"				);
	assert_m( out_swap_chain_data_pointer	!= NULL, "No swap chain data storage found"	);
	assert_m( swap_chain_image_views_pointer!= NULL, "No image views storage found"		);

	/* required by optional imageless frame buffer */
	out_swap_chain_data_pointer->frame_buffer			= VK_NULL_HANDLE;
	out_swap_chain_data_pointer->frame_buffers_pointer	= NULL;

	struct VkFramebufferCreateInfo frame_buffer_create_information = {
		.sType				= VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass			= application_pointer->render_pass,
		/* same with render_pass_create_information */
		.attachmentCount	= VSR_ATTACHMENT_COLOR_AMOUNT,
		.width				= out_swap_chain_data_pointer->extent.width,
		.height				= out_swap_chain_data_pointer->extent.height,
		.layers				= 1
	};

	if(	application_pointer->capabilities_device.has_imageless_frame_buffer == false )
		return vsr_frame_buffer_create_imaged(
			application_pointer, swap_chain_image_views_pointer,
			out_swap_chain_data_pointer, frame_buffer_create_information
		);

	VkFramebufferAttachmentImageInfoKHR attachment_image_information = {
		.sType				= VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO_KHR,
		.usage				= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.width				= out_swap_chain_data_pointer->extent.width,
		.height				= out_swap_chain_data_pointer->extent.height,
		.layerCount			= 1,
		.viewFormatCount	= 1,
		.pViewFormats		= &out_swap_chain_data_pointer->image_format
	};
	VkFramebufferAttachmentsCreateInfoKHR attachment_create_information = {
		.sType						= VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO_KHR,
		.attachmentImageInfoCount	= 1,
		.pAttachmentImageInfos		= &attachment_image_information
	};
	frame_buffer_create_information.pNext = &attachment_create_information;
	frame_buffer_create_information.flags = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT_KHR;

	return vkCreateFramebuffer_wrapped(
				application_pointer->device, &frame_buffer_create_information, NULL,
				&out_swap_chain_data_pointer->frame_buffer
			) == VK_SUCCESS
		? NULL
		: "(vsr_frame_buffer_create) frame buffer creation failed";
}

static void vsr_frame_buffer_destroy(
		VkDevice device, VkFramebuffer frame_buffer,
		VkFramebuffer * restrict frame_buffers_pointer, uint32_t frame_buffer_amount
	)
{
	assert_m( device != VK_NULL_HANDLE, "No device found" );

	if(	frame_buffer != VK_NULL_HANDLE ) {
		vkDestroyFramebuffer( device, frame_buffer, NULL );
		return;
	}

	assert_m( frame_buffers_pointer != NULL, "No frame buffers storage found" );
	for(uint32_t frame_buffer_index = 0;
			frame_buffer_index < frame_buffer_amount;
				++frame_buffer_index )
		vkDestroyFramebuffer( device, frame_buffers_pointer[frame_buffer_index], NULL );

	free( frame_buffers_pointer );
}

static const char * vsr_render_pass_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	struct VkAttachmentDescription color_attachment = {
		.format			= application_pointer->swap_chain_data.image_format,
		.samples		= VK_SAMPLE_COUNT_1_BIT,
		.loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp		= VK_ATTACHMENT_STORE_OP_STORE,
		.stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		.stencilStoreOp	= VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED,
		.finalLayout	= VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
	};

	struct VkAttachmentReference color_attachment_reference = {
		.layout		= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
	};

	struct VkSubpassDescription subpass = {
		.pipelineBindPoint		= VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount	= VSR_ATTACHMENT_COLOR_AMOUNT,
		.pColorAttachments		= &color_attachment_reference
	};

	struct VkSubpassDependency dependency = {
		.srcSubpass		= VK_SUBPASS_EXTERNAL,
		.srcStageMask	= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstStageMask	= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstAccessMask	= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
	};

	struct VkRenderPassCreateInfo render_pass_create_information = {
		.sType				= VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount	= VSR_ATTACHMENT_COLOR_AMOUNT,
		.pAttachments		= &color_attachment,
		.subpassCount		= 1,
		.pSubpasses			= &subpass,
		.dependencyCount	= 1,
		.pDependencies		= &dependency
	};

	return( vkCreateRenderPass_wrapped(
				application_pointer->device, &render_pass_create_information, NULL,
				&application_pointer->render_pass
			) == VK_SUCCESS )
		? NULL
		: "(vsr_render_pass_create) render pass creation failed";
}

static const char * vsr_graphics_pipeline_from_shaders_create(
		struct VSR_Application * restrict application_pointer,
		const VkShaderModule shader_module_vertex, const VkShaderModule shader_module_fragment
	)
{
	assert_m( application_pointer	!= NULL,			"No application found"				);
	assert_m( shader_module_vertex	!= VK_NULL_HANDLE,	"No vertex shader module found"		);
	assert_m( shader_module_fragment!= VK_NULL_HANDLE,	"No fragment shader module found"	);

	VkPipelineShaderStageCreateInfo shader_stages_information[] = {
		{
			.sType	= VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage	= VK_SHADER_STAGE_VERTEX_BIT,
			.module	= shader_module_vertex,
			.pName	= "main"
		},
		{
			.sType	= VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage	= VK_SHADER_STAGE_FRAGMENT_BIT,
			.module	= shader_module_fragment,
			.pName	= "main"
		}
	};

	/* data to ignore in configuration, must be specified in drawing time */
	const VkDynamicState dynamics_states_array[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};
	const uint32_t dynamics_states_amount =
		sizeof(dynamics_states_array) / sizeof(VkDynamicState);

	struct VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType				= VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount	= dynamics_states_amount,
		.pDynamicStates		= dynamics_states_array
	};

	struct VkVertexInputBindingDescription binding_description = vsr_get_binding_description();

	const struct VkVertexInputAttributeDescription attribute_descriptions_array[2] = {
		{
			0 ,0, VK_FORMAT_R32G32_SFLOAT, offsetof(struct VSR_Vertex, position)
		}, {
			1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(struct VSR_Vertex, color)
		}
	};
	const uint32_t attribute_descriptions_amount =
		sizeof(attribute_descriptions_array) / sizeof(*attribute_descriptions_array);


	struct VkPipelineVertexInputStateCreateInfo vertex_input_information = {
		.sType						= VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount		= 1,
		.pVertexBindingDescriptions			= &binding_description,
		.vertexAttributeDescriptionCount	= attribute_descriptions_amount,
		.pVertexAttributeDescriptions		= attribute_descriptions_array
	};

	struct VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType		= VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology	= VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	/* multiple viewports require GPU feature (logical device creation) */
	struct VkViewport viewport = {
		.width		= (float) application_pointer->swap_chain_data.extent.width,
		.height		= (float) application_pointer->swap_chain_data.extent.height,
		.maxDepth	= 1.0f
	};

	struct VkRect2D scissor = {
		.offset	= { 0, 0 },
		.extent	= application_pointer->swap_chain_data.extent
	};

	struct VkPipelineViewportStateCreateInfo viewport_state = {
		.sType			= VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount	= 1,
		.pViewports		= &viewport,
		.scissorCount	= 1,
		.pScissors		= &scissor
	};

	struct VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType					= VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode				= VK_POLYGON_MODE_FILL,
		.cullMode					= VK_CULL_MODE_BACK_BIT,
		.frontFace					= VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth					= 1.0f
	};

	struct VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType					= VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples	= VK_SAMPLE_COUNT_1_BIT,
		.minSampleShading		= 1.0f
	};

	/* per frame_buffer structure */
	struct VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable			= VK_FALSE,
		.srcColorBlendFactor	= VK_BLEND_FACTOR_ONE,
		.dstColorBlendFactor	= VK_BLEND_FACTOR_ZERO,
		.colorBlendOp			= VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor	= VK_BLEND_FACTOR_ONE,
		.dstAlphaBlendFactor	= VK_BLEND_FACTOR_ZERO,
		.alphaBlendOp			= VK_BLEND_OP_ADD,
		.colorWriteMask			=
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
	};

	struct VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType				= VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOp			= VK_LOGIC_OP_COPY,
		.attachmentCount	= 1,
		.pAttachments		= &color_blend_attachment
	};

	struct VkPipelineLayoutCreateInfo pipeline_layout_information = {
		.sType					= VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount			= 1,
		.pSetLayouts			= &application_pointer->descriptor_set_layout
	};

	if(	vkCreatePipelineLayout_wrapped(
			application_pointer->device, &pipeline_layout_information, NULL,
			&application_pointer->pipeline_layout
		) != VK_SUCCESS )
		return "(vsr_graphics_pipeline_from_shaders_create) pipeline layout creation failed";

	struct VkGraphicsPipelineCreateInfo pipeline_create_information =
		(struct VkGraphicsPipelineCreateInfo)
	{
		.sType					= VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount				= 2,
		.pStages				= shader_stages_information,
		.pVertexInputState		= &vertex_input_information,
		.pInputAssemblyState	= &input_assembly,
		.pViewportState			= &viewport_state,
		.pRasterizationState	= &rasterizer,
		.pMultisampleState		= &multisampling,
		.pColorBlendState		= &color_blending,
		.pDynamicState			= &dynamic_state,
		.layout					= application_pointer->pipeline_layout,
		.renderPass				= application_pointer->render_pass,
		.basePipelineHandle		= VK_NULL_HANDLE,
		.basePipelineIndex		= -1
	};

	if(	vkCreateGraphicsPipelines(
			application_pointer->device, VK_NULL_HANDLE, 1, &pipeline_create_information, NULL,
			&application_pointer->graphics_pipeline
		) != VK_SUCCESS )
	{
		vkDestroyPipelineLayout(
			application_pointer->device, application_pointer->pipeline_layout, NULL
		);
		application_pointer->pipeline_layout = VK_NULL_HANDLE;
		return "(vsr_graphics_pipeline_from_shaders_create) graphics pipeline creation failed";
	}

	return NULL;
}

static const char * vsr_graphics_pipeline_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );
	const char * error_message_pointer;
	VkShaderModule shader_module_vertex, shader_module_fragment;

	size_t file_size = 0;
	char * shader_code_vertex = NULL, * shader_code_fragment = NULL;

	if(	hf_file_read( "shaders/vertex.spv", &shader_code_vertex, &file_size ) > 0 )
		return "(vsr_graphics_pipeline_create) vertex shader code failed to get";

	if((error_message_pointer = vsr_shader_module_create(
			application_pointer->device, shader_code_vertex, file_size, &shader_module_vertex
		)) != NULL )
		return error_message_pointer;

	if(	hf_file_read( "shaders/fragment.spv", &shader_code_fragment, &file_size ) > 0 ) {
		error_message_pointer =
			"(vsr_graphics_pipeline_create) fragment shader code failed to get";
		goto cleanup_vertex;
	}

	if((error_message_pointer = vsr_shader_module_create(
			application_pointer->device, shader_code_fragment, file_size, &shader_module_fragment
		)) != NULL )
		goto cleanup_vertex;

	error_message_pointer = vsr_graphics_pipeline_from_shaders_create(
		application_pointer, shader_module_vertex, shader_module_fragment
	);

	vkDestroyShaderModule( application_pointer->device, shader_module_fragment, NULL );

cleanup_vertex:
	vkDestroyShaderModule( application_pointer->device, shader_module_vertex, NULL );

	return error_message_pointer;
}

static const char * vsr_shader_module_create(
		VkDevice device, char * restrict shader_code_source_pointer, size_t file_size,
		VkShaderModule * restrict out_shader_module_pointer
	)
{
	assert_m( device					!= VK_NULL_HANDLE,	"No device found"				);
	assert_m( shader_code_source_pointer!= NULL,			"No shader source found"		);
	assert_m( file_size					> 0,				"No file size found"			);
	assert_m( out_shader_module_pointer	!= NULL,			"No shader module storage found");

	if(	file_size % sizeof(uint32_t) != 0 ) {
		free( shader_code_source_pointer );
		return
			"(vsr_shader_module_create) "
			"file size is not a multiple of 4, SPIR-V shader file is corrupted";
	}

	uint32_t * aligned_shader_source_code_pointer;
	if(	am_aligned_malloc(
			&aligned_shader_source_code_pointer, sizeof(*aligned_shader_source_code_pointer),
			file_size
		) == false )
	{
		free( shader_code_source_pointer );
		return "(vsr_shader_module_create) buffer allocation cause overflow";
	} else if ( aligned_shader_source_code_pointer == NULL ) {
		free( shader_code_source_pointer );
		return "(vsr_shader_module_create) aligned code failed to allocate";
	}

	memcpy( aligned_shader_source_code_pointer, shader_code_source_pointer, file_size );
	free( shader_code_source_pointer );

	if(	aligned_shader_source_code_pointer[0] != VSR_SPIRV_MAGIC_RECOGNITION_NUMBER ) {
		am_aligned_free( aligned_shader_source_code_pointer );
		return "(vsr_shader_module_create) SPIR-V file detection failed";
	}

	VkShaderModuleCreateInfo create_information = {
		.sType		= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize	= file_size,
		.pCode		= aligned_shader_source_code_pointer
	};

	VkResult result = vkCreateShaderModule(
		device, &create_information, NULL, out_shader_module_pointer
	);
	am_aligned_free( aligned_shader_source_code_pointer );

	return (result != VK_SUCCESS)
		? "(vsr_shader_module_create) shader module creation failed"
		: NULL;
}

static const char * vsr_image_views_create(
		struct VSR_Application * restrict application_pointer,
		VkImage * restrict swap_chain_images_pointer, uint32_t swap_chain_image_views_amount,
		VkFormat swap_chain_image_format, VkImageView ** out_swap_chain_image_views_pointer
	)
{
	assert_m( application_pointer				!= NULL, "No application found"			);
	assert_m( swap_chain_images_pointer			!= NULL, "No swap chain storage found"	);
	assert_m( out_swap_chain_image_views_pointer!= NULL, "No images storage found"		);

	if(	sa_malloc_array(
			out_swap_chain_image_views_pointer, swap_chain_image_views_amount,
			sizeof(**out_swap_chain_image_views_pointer)
		) == false )
		return "(vsr_image_views_create) allocation size overflow";
	else if ( *out_swap_chain_image_views_pointer == NULL )
		return "(vsr_image_views_create) image views memory allocation failed";

	for ( size_t image_index = 0; image_index < swap_chain_image_views_amount; ++image_index ) {
		VkImageViewCreateInfo create_information = {
			.sType		= VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image		= swap_chain_images_pointer[image_index],
			.viewType	= VK_IMAGE_VIEW_TYPE_2D,
			.format		= swap_chain_image_format,
			.components	= {
				.r		= VK_COMPONENT_SWIZZLE_IDENTITY,
				.g		= VK_COMPONENT_SWIZZLE_IDENTITY,
				.b		= VK_COMPONENT_SWIZZLE_IDENTITY,
				.a		= VK_COMPONENT_SWIZZLE_IDENTITY
			},
			.subresourceRange = {
				.aspectMask			= VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount			= 1,
				.layerCount			= 1
			}
		};
		if(vkCreateImageView(
				application_pointer->device, &create_information, NULL,
				&(*out_swap_chain_image_views_pointer)[image_index]
			) != VK_SUCCESS )
		{
			vsr_image_views_destroy(
				application_pointer->device, *out_swap_chain_image_views_pointer,
				(uint32_t) image_index
			);
			*out_swap_chain_image_views_pointer = NULL;
			return "(vsr_image_views_create) image views creation failed";
		}
	}

	return NULL;
}

static inline const char * vsr_swap_chain_support_check(
		const struct VSR_Swap_Chain_Support_Details * restrict swap_chain_support_pointer,
		const uint32_t image_dimension_2d_maximal
	)
{
	const struct VkExtent2D image_extension_maximal =
		swap_chain_support_pointer->surface_capabilities.maxImageExtent;

	if(	image_extension_maximal.width == 0 ||
		image_extension_maximal.height== 0 )
		return "(vsr_swap_chain_support_check) zero surface extent";

	if(	image_extension_maximal.width > image_dimension_2d_maximal ||
		image_extension_maximal.height> image_dimension_2d_maximal )
		return "(vsr_swap_chain_support_check) maximal image extension exceeds device limits";

	return
		swap_chain_support_pointer->formats_amount		== 0 ||
		swap_chain_support_pointer->present_modes_amount== 0
		? "(vsr_swap_chain_support_check) no surface formats or present modes found"
		: NULL;
}

static const char * vsr_swap_chain_create(
		struct VSR_Application * restrict application_pointer,
		struct VSR_Swap_Chain_Data * restrict out_swap_chain_data_pointer,
		VkImage ** restrict out_swap_chain_images_pointer
	)
{
	assert_m( application_pointer			!= NULL, "No application found"					);
	assert_m( out_swap_chain_data_pointer	!= NULL, "No swap chain data storage found"		);
	assert_m( out_swap_chain_images_pointer	!= NULL, "No swap chain images storage found"	);

	/* nullified to make free simple */
	struct VSR_Swap_Chain_Support_Details swap_chain_support = { 0 };
	const char * error_message_pointer;
	if((error_message_pointer = vsr_swap_chain_support_query(
			application_pointer->surface, application_pointer->device_physical,
			&swap_chain_support
		)) != NULL )
		return error_message_pointer;

	if((error_message_pointer = vsr_swap_chain_support_check(
			&swap_chain_support, application_pointer->image_dimension_2d_maximal
		)) != NULL ) {
		vsr_swap_chain_support_details_free( &swap_chain_support );
		return error_message_pointer;
	}

	struct VkSurfaceFormatKHR surface_format = vsr_swap_surface_format_choose(
		swap_chain_support.surface_formats_pointer, swap_chain_support.formats_amount
	);

	vsr_swap_chain_extent_write(
		application_pointer,
		&out_swap_chain_data_pointer->extent, &swap_chain_support.surface_capabilities
	);

	if(	out_swap_chain_data_pointer->extent.width == 0 ||
		out_swap_chain_data_pointer->extent.height== 0 )
	{
		vsr_swap_chain_support_details_free( &swap_chain_support );
		return "(vsr_swap_chain_create) zero extent";
	}

	out_swap_chain_data_pointer->image_format = surface_format.format;
	out_swap_chain_data_pointer->image_views_amount =
			(swap_chain_support.surface_capabilities.maxImageCount == 0)
		? swap_chain_support.surface_capabilities.minImageCount + 1
		: (uint32_t) cv_clamp_int64_t(
			(uint64_t) swap_chain_support.surface_capabilities.minImageCount + 1,
			(uint64_t) swap_chain_support.surface_capabilities.minImageCount,
			(uint64_t) swap_chain_support.surface_capabilities.maxImageCount);

	VkSwapchainPresentScalingCreateInfoEXT scaling_create_information = {
		.sType				= VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT,
		.scalingBehavior	= VK_PRESENT_SCALING_STRETCH_BIT_EXT
	};

	VkSwapchainKHR old_swap_chain = application_pointer->swap_chain_data.swap_chain;
	struct VkSwapchainCreateInfoKHR create_information = {
		.sType				= VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.pNext				=
			(application_pointer->capabilities_device.has_present_scaling_stretch == true)
				? &scaling_create_information
				: NULL,
		.surface			= application_pointer->surface,
		.minImageCount		= out_swap_chain_data_pointer->image_views_amount,
		.imageFormat		= surface_format.format,
		.imageColorSpace	= surface_format.colorSpace,
		.imageExtent		= out_swap_chain_data_pointer->extent,
		.imageArrayLayers	= 1,/* image consists of this amount of layers */
		.imageUsage			= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.preTransform		= swap_chain_support.surface_capabilities.currentTransform,
		.compositeAlpha		= VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode		= VK_PRESENT_MODE_FIFO_KHR,
		.clipped			= VK_TRUE,
		.oldSwapchain		= old_swap_chain
	};
	vsr_swap_chain_support_details_free( &swap_chain_support );

	uint32_t unique_families_array[] = {
		application_pointer->queue_family_indices.graphics_family,
		application_pointer->queue_family_indices.present_family
	};
	if(	application_pointer->queue_family_indices.graphics_family !=
		application_pointer->queue_family_indices.present_family )
	{
		create_information.imageSharingMode			= VK_SHARING_MODE_CONCURRENT;
		create_information.queueFamilyIndexCount	= 2;
		create_information.pQueueFamilyIndices		= unique_families_array;
	} else {
		create_information.imageSharingMode			= VK_SHARING_MODE_EXCLUSIVE;
	}

	VkResult result = vkCreateSwapchainKHR_wrapped(
		application_pointer->device, &create_information, NULL,
		&out_swap_chain_data_pointer->swap_chain
	);

	/* old swap chain is retired even on fall, trying to create a new one from scratch */
	if(	result == VK_ERROR_NATIVE_WINDOW_IN_USE_KHR &&
		create_information.oldSwapchain != VK_NULL_HANDLE )
	{
		create_information.oldSwapchain = VK_NULL_HANDLE;
		result = vkCreateSwapchainKHR_wrapped(
			application_pointer->device, &create_information, NULL,
			&out_swap_chain_data_pointer->swap_chain
		);
	}

	if(	result != VK_SUCCESS )
		return "(vsr_swap_chain_create) swap chain failed to create";

	if(	vkGetSwapchainImagesKHR(
			application_pointer->device, out_swap_chain_data_pointer->swap_chain,
			&out_swap_chain_data_pointer->image_views_amount, NULL
		) != VK_SUCCESS )
	{
		error_message_pointer = "(vsr_swap_chain_create) image count for swap chain failed to get";
		goto cleanup;
	} else if ( out_swap_chain_data_pointer->image_views_amount == 0 ) {
		error_message_pointer = "(vsr_swap_chain_create) no images returned";
		goto cleanup;
	}

	if(	sa_malloc_array(
			out_swap_chain_images_pointer,
			out_swap_chain_data_pointer->image_views_amount,
			sizeof(**out_swap_chain_images_pointer)
		) == false )
	{
		error_message_pointer = "(vsr_swap_chain_create) allocation size overflow";
		goto cleanup;
	} else if ( *out_swap_chain_images_pointer == NULL ) {
		error_message_pointer =
			"(vsr_swap_chain_create) swap chain images memory allocation failed";
		goto cleanup;
	}

	if(	vkGetSwapchainImagesKHR(
			application_pointer->device, out_swap_chain_data_pointer->swap_chain,
			&out_swap_chain_data_pointer->image_views_amount,
			*out_swap_chain_images_pointer
		) != VK_SUCCESS )
	{
		error_message_pointer = "(vsr_swap_chain_create) swap chain images failed to create";
		goto cleanup_images;
	}

	return NULL;

cleanup_images:
	free( *out_swap_chain_images_pointer );
	*out_swap_chain_images_pointer = NULL;

cleanup:
	vkDestroySwapchainKHR(
		application_pointer->device, out_swap_chain_data_pointer->swap_chain, NULL
	);
	out_swap_chain_data_pointer->swap_chain = VK_NULL_HANDLE;
	return error_message_pointer;
}

static inline void vsr_swap_chain_extent_write(
		struct VSR_Application * restrict application_pointer,
		struct VkExtent2D * restrict out_extent_pointer,
		struct VkSurfaceCapabilitiesKHR * restrict out_surface_capabilities_pointer
	)
{
	pthread_mutex_lock(&application_pointer->render_mutex);
	uint32_t frame_buffer_width = (uint32_t) application_pointer->frame_state.width;
	uint32_t frame_buffer_height= (uint32_t) application_pointer->frame_state.height;
	pthread_mutex_unlock(&application_pointer->render_mutex);

	*out_extent_pointer = vsr_swap_extent_choose(
		out_surface_capabilities_pointer, frame_buffer_width, frame_buffer_height
	);
}

static struct VkSurfaceFormatKHR vsr_swap_surface_format_choose(
		const struct VkSurfaceFormatKHR * restrict available_formats_pointer,
		size_t available_formats_amount
	)
{
	assert_m( available_formats_pointer	!= NULL,"No available formats list found"	);
	assert_m( available_formats_amount	> 0,	"No available formats found"		);

	for(size_t format_index = 0; format_index < available_formats_amount; ++format_index ) {
		if(available_formats_pointer[format_index].format == VK_FORMAT_B8G8R8A8_SRGB &&
			available_formats_pointer[format_index].
				colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			return available_formats_pointer[format_index];
	}
	return available_formats_pointer[0].format == VK_FORMAT_UNDEFINED
		? (struct VkSurfaceFormatKHR) {
				.format		= VK_FORMAT_B8G8R8A8_SRGB,
				.colorSpace	= available_formats_pointer[0].colorSpace
			}
		: available_formats_pointer[0];
}

static struct VkExtent2D vsr_swap_extent_choose(
		const struct VkSurfaceCapabilitiesKHR * restrict surface_capabilities_pointer,
		uint32_t frame_buffer_width, uint32_t frame_buffer_height
	)
{
	assert_m( surface_capabilities_pointer != NULL, "No surface capabilities list found" );

	if(	surface_capabilities_pointer->currentExtent.width != UINT32_MAX )
		return surface_capabilities_pointer->currentExtent;

	struct VkExtent2D extent = { frame_buffer_width, frame_buffer_height };

	extent.width = (uint32_t) cv_clamp_int64_t(
		(uint64_t) extent.width,
		(uint64_t) surface_capabilities_pointer->minImageExtent.width,
		(uint64_t) surface_capabilities_pointer->maxImageExtent.width);

	extent.height = (uint32_t) cv_clamp_int64_t(
		(uint64_t) extent.height,
		(uint64_t) surface_capabilities_pointer->minImageExtent.height,
		(uint64_t) surface_capabilities_pointer->maxImageExtent.height);

	return extent;
}

static bool vsr_instance_create( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	bool is_instance_created = false;

#ifndef NDEBUG

	global_is_validation_layer_supported = vsr_validation_layer_support_check();

#endif

	struct VkApplicationInfo application_information = {
		.sType				= VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName	= "Vulkan Square Rotating",
		.applicationVersion = VK_MAKE_VERSION( 1, 0, 0 ),
		.pEngineName		= "No Engine",
		.engineVersion		= VK_MAKE_VERSION( 1, 0, 0 ),
		.apiVersion			= VK_API_VERSION_1_0
	};

	struct VSR_Extension_Names_Mutable extensions;
	const char * error_message_pointer = vsr_instance_extensions_check(
		application_pointer, &extensions
	);
	if(	error_message_pointer != NULL ) {
		woem_push( "%s", error_message_pointer );
		return false;
	}

	struct VkInstanceCreateInfo create_information = {
		.sType						= VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo			= &application_information,
		.enabledExtensionCount		= extensions.amount,
		.ppEnabledExtensionNames	= extensions.data_pointer
	};

#ifndef NDEBUG

	struct VkDebugUtilsMessengerCreateInfoEXT debug_create_information;
	if(	global_is_validation_layer_supported == true ) {
		vsr_debug_messenger_create_information_populate( &debug_create_information );

		create_information.enabledLayerCount	= (uint32_t) global_validation_layers.amount;
		create_information.ppEnabledLayerNames	= global_validation_layers.data_pointer;
		create_information.pNext =
			(struct VkDebugUtilsMessengerCreateInfoEXT *) &debug_create_information;
	}
	else
		VSR_DEBUG_LOG("(vsr_instance_create) requested validation layers aren't available");

#endif

	is_instance_created = vkCreateInstance_wrapped(
		&create_information, NULL, &application_pointer->instance
	) == VK_SUCCESS;

	free( extensions.data_pointer );

	if(	is_instance_created == false ) {
		woem_push( "(vsr_instance_create) failed to create instance" );
		return false;
	}

	application_pointer->capabilities_vulkan.get_physical_device_features_2 =
		(PFN_vkGetPhysicalDeviceFeatures2KHR) vkGetInstanceProcAddr(
			application_pointer->instance, "vkGetPhysicalDeviceFeatures2KHR"
		);
	application_pointer->capabilities_vulkan.get_physical_device_surface_capabilities_2 =
		(PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR) vkGetInstanceProcAddr(
			application_pointer->instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR"
		);

	return true;
}

static const char * vsr_instance_extensions_check(
		struct VSR_Application * restrict application_pointer,
		struct VSR_Extension_Names_Mutable * restrict out_extensions_pointer
	)
{
	assert_m( application_pointer	!= NULL, "No application found"				);
	assert_m( out_extensions_pointer!= NULL, "No extensions names storage found");

	struct VSR_Extension_Properties_Mutable extensions_available_mutable;
	const char * error_message_pointer = vsr_get_extensions_available(
		&extensions_available_mutable
	);
	if(	error_message_pointer != NULL )
		return error_message_pointer;
	else if(extensions_available_mutable.amount == 0 )
		return "(vsr_instance_extensions_check) no instance extensions available";

	struct VSR_Extension_Names extensions_required;
	error_message_pointer = vsr_get_instance_extensions_required( &extensions_required );
	if(error_message_pointer != NULL ) {
		free( extensions_available_mutable.data_pointer );
		return error_message_pointer;
	}

	struct VSR_Extension_Properties extensions_available = vsr_extension_properties_freeze(
		extensions_available_mutable
	);
	if(	vsr_instance_extensions_required_check(extensions_required,extensions_available) == false)
	{
		free( extensions_available_mutable.data_pointer );
		return "(vsr_instance_extensions_check) mandatory instance extension missing";
	}

	struct VSR_Extension_Group instance_extension_groups_array
		[VSR_EXTENSION_GROUPS_AMOUNT_INSTANCE];
	vsr_instance_extensions_groups_fill(application_pointer, instance_extension_groups_array);
	vsr_extensions_groups_availability(
		(struct VSR_Extension_Groups) {
			.data_pointer	= instance_extension_groups_array,
			.amount			=
				sizeof(instance_extension_groups_array)/sizeof(*instance_extension_groups_array)
		},
		extensions_available
	);

	free( extensions_available_mutable.data_pointer );

	return vsr_instance_extensions_all_build(
		extensions_required, instance_extension_groups_array, out_extensions_pointer
	);
}

static bool vsr_device_capabilities_build(
		VkPhysicalDevice device, VkSurfaceKHR surface,
		const struct VSR_Capabilities_Vulkan * restrict instance_capabilities_pointer,
		struct VSR_Capabilities_Device * restrict out_capabilities_device_pointer
	)
{
	assert_m(instance_capabilities_pointer	!= NULL, "No instance capabilities found"		);
	assert_m(out_capabilities_device_pointer!= NULL, "No device capabilities storage found"	);

	struct VkExtensionProperties extensions_array[VSR_LIMIT_STACK_EXTENSIONS];
	struct VSR_Extension_Properties_Mutable extensions_data;

	if(	vsr_device_extensions_get(device, extensions_array, &extensions_data) == false )
		return false;

	struct VSR_Extension_Properties extensions_available = vsr_extension_properties_freeze(
		extensions_data
	);
	bool has_extensions_required = vsr_extensions_available_check(
		global_device_extensions_required, extensions_available
	);

	if(	has_extensions_required == true ) {
		struct VSR_Extension_Group groups_extension_array[VSR_EXTENSION_GROUPS_AMOUNT_DEVICE];
		vsr_device_extensions_groups_fill(
			out_capabilities_device_pointer, groups_extension_array
		);
		vsr_extensions_groups_availability(
			(struct VSR_Extension_Groups) {
				.data_pointer	= groups_extension_array,
				.amount			= sizeof(groups_extension_array)/sizeof(*groups_extension_array)
			},
			extensions_available
		);

		if(	instance_capabilities_pointer->has_get_physical_device_properties_2 == true ) {
			struct VSR_Capabilities_Device features_device;
			vsr_device_extension_required_check(
				device, instance_capabilities_pointer->get_physical_device_features_2,
				&features_device
			);
			if(	instance_capabilities_pointer->has_surface_maintenance_1		== false ||
				features_device.has_swapchain_maintenance_1						== false )
			{
				out_capabilities_device_pointer->has_swapchain_maintenance_1	= false;
				out_capabilities_device_pointer->has_present_scaling_stretch	= false;
			}
			else {
				out_capabilities_device_pointer->has_present_scaling_stretch	=
					vsr_device_present_scaling_stretch_check(
						device, surface, instance_capabilities_pointer
					);
			}

			if(	features_device.has_imageless_frame_buffer						== false )
				out_capabilities_device_pointer->has_imageless_frame_buffer		= false;
		} else {
			*out_capabilities_device_pointer = (struct VSR_Capabilities_Device) { 0 };
		}
	}

	if(	extensions_data.data_pointer != extensions_array )
		free( extensions_data.data_pointer );

	return has_extensions_required;
}

static inline bool vsr_instance_extensions_required_check(
		const struct VSR_Extension_Names		required,
		const struct VSR_Extension_Properties	available
	)
{
	assert_m(
		required.amount == 0 || required.data_pointer != NULL, "No extension names list found"
	);
	assert_m(
		available.amount== 0 || available.data_pointer!= NULL,"No available extensions list found"
	);

#ifndef NDEBUG

	VSR_DEBUG_LOG("required extensions:");
	for(uint32_t extension_required_index = 0;
			extension_required_index < required.amount; ++extension_required_index )
		VSR_DEBUG_LOGF("\t%s", required.data_pointer[extension_required_index] );

	VSR_DEBUG_LOG("\navailable extensions:");
	for(uint32_t extension_index = 0; extension_index < available.amount; ++extension_index )
		VSR_DEBUG_LOGF( "\t%s", available.data_pointer[extension_index].extensionName );

#endif

	return vsr_extensions_available_check(required, available);
}

static inline bool vsr_extensions_available_check(
		const struct VSR_Extension_Names		required,
		const struct VSR_Extension_Properties	available
	)
{
	assert_m(
		required.amount == 0 || required.data_pointer != NULL, "No extension names list found"
	);
	assert_m(
		available.amount== 0 || available.data_pointer!= NULL,"No available extensions list found"
	);
	for(uint32_t extension_index_check = 0;
			extension_index_check < required.amount; ++extension_index_check )
	{
		assert_m(
			required.data_pointer[extension_index_check] != NULL, "No extension name found"
		);
		bool found = false;
		for(uint32_t extension_index_available = 0;
				extension_index_available < available.amount; ++extension_index_available)
		{
			if(	strcmp(
					required.data_pointer[extension_index_check],
					available.data_pointer[extension_index_available].extensionName
				) == 0 )
			{
				found = true;
				break;
			}
		}
		if(	found == false ) {
			VSR_DEBUG_LOGF(
				"(vsr_extensions_available_check) Warning: missing extension: %s",
				required.data_pointer[extension_index_check]
			);
			return false;
		}
	}
	return true;
}

static inline struct VSR_Extension_Properties vsr_extension_properties_freeze(
		const struct VSR_Extension_Properties_Mutable array_extension_properties_mutable
	)
{
	return (struct VSR_Extension_Properties) {
		.data_pointer	= array_extension_properties_mutable.data_pointer,
		.amount			= array_extension_properties_mutable.amount
	};
}

static const char * vsr_get_instance_extensions_required(
		struct VSR_Extension_Names * restrict extensions_required_pointer
	)
{
	assert_m( extensions_required_pointer != NULL, "No extensions data storage found" );

	extensions_required_pointer->data_pointer = (const char * const *)
		glfwGetRequiredInstanceExtensions(&extensions_required_pointer->amount);

	return (extensions_required_pointer->data_pointer == NULL)
		? "(vsr_get_instance_extensions_required) getting required instance extensions failed"
		: NULL;
}

static const char * vsr_get_extensions_available(
		struct VSR_Extension_Properties_Mutable * restrict out_extensions_pointer
	)
{
	assert_m( out_extensions_pointer != NULL, "No extensions storage found" );

	if(	vkEnumerateInstanceExtensionProperties(
			NULL, &out_extensions_pointer->amount, NULL
		) != VK_SUCCESS )
		return
			"(vsr_get_extensions_available) "
			"Vulkan Enumeration the Number of Instance Extension Properties failed";

	if(	out_extensions_pointer->amount == 0 )
		return NULL;

	if(	sa_malloc_array(
			&out_extensions_pointer->data_pointer, out_extensions_pointer->amount,
			sizeof(*out_extensions_pointer->data_pointer)
		) == false )
		return "(vsr_get_extensions_available) allocation size overflow";
	else if ( out_extensions_pointer->data_pointer == NULL )
		return "(vsr_get_extensions_available) Vulkan Extension Properties allocation failed";

	if(	vkEnumerateInstanceExtensionProperties(
			NULL, &out_extensions_pointer->amount, out_extensions_pointer->data_pointer
		) != VK_SUCCESS )
	{
		free( out_extensions_pointer->data_pointer );
		return
			"(vsr_get_extensions_available) "
			"Vulkan Enumeration of Instance Extension Properties failed";
	}

	return NULL;
}

static void vsr_extensions_groups_availability(
		const struct VSR_Extension_Groups		groups,
		const struct VSR_Extension_Properties	extensions_available
	)
{
	assert_m(extensions_available.data_pointer != NULL, "No available extensions found");

	for( uint32_t group_index = 0; group_index < groups.amount; ++group_index )
		*groups.data_pointer[group_index].is_available_pointer =
			vsr_extensions_available_check(
				groups.data_pointer[group_index].names, extensions_available
			);
}

static const char * vsr_instance_extensions_all_build(
		const struct VSR_Extension_Names extensions_required,
		const struct VSR_Extension_Group out_groups_extension_array[
			static VSR_EXTENSION_GROUPS_AMOUNT_INSTANCE
		],
		struct VSR_Extension_Names_Mutable * restrict out_extensions_pointer
	)
{
	out_extensions_pointer->amount = extensions_required.amount;
	for(uint32_t group_index = 0;
			group_index < VSR_EXTENSION_GROUPS_AMOUNT_INSTANCE;
				++group_index )
		if(*out_groups_extension_array[group_index].is_available_pointer == true)
			out_extensions_pointer->amount +=
				out_groups_extension_array[group_index].names.amount;

#ifndef NDEBUG

	if(	global_is_validation_layer_supported == true )
		++out_extensions_pointer->amount;

#endif

	if(	sa_malloc_array(
			&out_extensions_pointer->data_pointer, out_extensions_pointer->amount,
			sizeof(*out_extensions_pointer->data_pointer)
		) == false )
		return "(vsr_instance_extensions_all_build) allocation size overflow";
	else if ( out_extensions_pointer->data_pointer == NULL )
		return "(vsr_instance_extensions_all_build) extensions array allocation failed";

	memcpy(
		out_extensions_pointer->data_pointer,
		extensions_required.data_pointer,
		extensions_required.amount * sizeof(*out_extensions_pointer->data_pointer)
	);

	out_extensions_pointer->amount = extensions_required.amount;
	for( uint32_t group_index = 0;
			group_index < VSR_EXTENSION_GROUPS_AMOUNT_INSTANCE;
				++group_index )
	{
		if(	*out_groups_extension_array[group_index].is_available_pointer == false )
			continue;
		memcpy(
			out_extensions_pointer->data_pointer + out_extensions_pointer->amount,
			out_groups_extension_array[group_index].names.data_pointer,
			out_groups_extension_array[group_index].names.amount *
				sizeof(*out_extensions_pointer->data_pointer)
		);
		out_extensions_pointer->amount += out_groups_extension_array[group_index].names.amount;
	}

#ifndef NDEBUG

	if(	global_is_validation_layer_supported == true )
		out_extensions_pointer->data_pointer[out_extensions_pointer->amount++] =
			VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

#endif

	return NULL;
}

static void vsr_device_extensions_groups_fill(
		struct VSR_Capabilities_Device * restrict capabilities_device_pointer,
		struct VSR_Extension_Group out_groups_extension_array[
			static VSR_EXTENSION_GROUPS_AMOUNT_DEVICE
		]
	)
{
	assert_m( capabilities_device_pointer != NULL, "No capabilities storage found" );

	out_groups_extension_array[0] = (struct VSR_Extension_Group) {
		.names					= global_device_extensions_swapchain_maintenance_1,
		.is_available_pointer	= &capabilities_device_pointer->has_swapchain_maintenance_1
	};
	out_groups_extension_array[1] = (struct VSR_Extension_Group) {
		.names					= global_device_extensions_imageless_frame_buffer,
		.is_available_pointer	= &capabilities_device_pointer->has_imageless_frame_buffer
	};
}

static void vsr_instance_extensions_groups_fill(
		struct VSR_Application * restrict application_pointer,
		struct VSR_Extension_Group out_groups_extension_array[
			static VSR_EXTENSION_GROUPS_AMOUNT_INSTANCE
		]
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	out_groups_extension_array[0] = (struct VSR_Extension_Group) {
		.names					= global_instance_extensions_device_properties,
		.is_available_pointer	= &application_pointer->capabilities_vulkan.
			has_get_physical_device_properties_2
	};
	out_groups_extension_array[1] = (struct VSR_Extension_Group) {
		.names					= global_instance_extensions_surface_capabilities,
		.is_available_pointer	= &application_pointer->capabilities_vulkan.
			has_surface_maintenance_1
	};
}

#ifndef NDEBUG
static bool vsr_validation_layer_support_check(void) {
	if(	global_validation_layers.amount == 0 )
		return true;

	uint32_t instance_layers_amount;
	VkResult result = vkEnumerateInstanceLayerProperties( &instance_layers_amount, NULL );
	if(	result != VK_SUCCESS )
		return false;

	struct VkLayerProperties * available_layers_pointer;
	if(	sa_malloc_array(
			&available_layers_pointer, instance_layers_amount, sizeof(*available_layers_pointer)
		) == false )
	{
		woem_push( "(vsr_validation_layer_support_check) allocation size overflow" );
		return false;
	}
	else if ( available_layers_pointer == NULL ) {
		woem_push( "(vsr_validation_layer_support_check) allocation failed" );
		return false;
	}

	result = vkEnumerateInstanceLayerProperties(
		&instance_layers_amount, available_layers_pointer
	);
	if(	result != VK_SUCCESS ) {
		free( available_layers_pointer );
		return false;
	}
	for ( uint32_t validation_layer_index = 0;
			validation_layer_index < global_validation_layers.amount; ++validation_layer_index ) {
		bool validation_layer_found = false;
		for ( uint32_t instance_layer_index = 0;
				instance_layers_amount > instance_layer_index; ++instance_layer_index )
		{
			if(	strcmp(
					global_validation_layers.data_pointer[validation_layer_index],
					available_layers_pointer[instance_layer_index].layerName
				) == 0 )
			{
				validation_layer_found = true;
				break;
			}
		}
		if(	validation_layer_found == false ) {
			free( available_layers_pointer );
			return false;
		}
	}
	free( available_layers_pointer );
	return true;
}

static VkResult vsr_debug_utils_messenger_extension_create(
		VkInstance instance,
		const VkDebugUtilsMessengerCreateInfoEXT * restrict create_information_pointer,
		VkDebugUtilsMessengerEXT * restrict debug_messenger_pointer
	)
{
	assert_m( create_information_pointer!= NULL,"No debug messenger creation information found"	);
	assert_m( debug_messenger_pointer	!= NULL,"No debug messenger found"						);

	PFN_vkCreateDebugUtilsMessengerEXT function = (PFN_vkCreateDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr( instance, "vkCreateDebugUtilsMessengerEXT" );
	if(	function != NULL )
		return function(
			instance, create_information_pointer, NULL, debug_messenger_pointer
		);
	else
		return VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void vsr_debug_utils_messenger_extension_destroy(
		VkInstance instance, VkDebugUtilsMessengerEXT debug_messenger
	)
{
	PFN_vkDestroyDebugUtilsMessengerEXT function_pointer = (PFN_vkDestroyDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr( instance, "vkDestroyDebugUtilsMessengerEXT" );
	if(	function_pointer != NULL )
		function_pointer( instance, debug_messenger, NULL );
}

static bool vsr_debug_messenger_setup( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	VkDebugUtilsMessengerCreateInfoEXT create_information;
	vsr_debug_messenger_create_information_populate( &create_information );
	return vsr_debug_utils_messenger_extension_create(
		application_pointer->instance, &create_information,
		&application_pointer->debug_messenger_function
	) == VK_SUCCESS;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL vsr_debug_callback_function(
		VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
		VkDebugUtilsMessageTypeFlagsEXT message_type,
		const VkDebugUtilsMessengerCallbackDataEXT * restrict data_callback_pointer,
		void * restrict data_user_pointer
	)
{
	(void) message_severity; (void) message_type; (void) data_user_pointer;
	fprintf( stderr, "validation layer: %s\n", data_callback_pointer->pMessage );
	return VK_FALSE;
}

static void vsr_debug_messenger_create_information_populate(
		struct VkDebugUtilsMessengerCreateInfoEXT * restrict creation_information_pointer
	)
{
	assert_m(
		creation_information_pointer != NULL, "No debug messenger creation information found"
	);

	*creation_information_pointer = (struct VkDebugUtilsMessengerCreateInfoEXT) {
		.sType =
			VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		.messageSeverity =
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT	|
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT	|
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
		.messageType =
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT		|
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT	|
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
		.pfnUserCallback = vsr_debug_callback_function
	};
}

static void vsr_debug_gpu_print(
		VkPhysicalDeviceProperties device_properties,
		struct VSR_Capabilities_Device capabilities_device, uint32_t scores
	)
{
	VSR_DEBUG_LOGF(
		"\nGPU candidate: %s \ntype: %s \nidentification number %d\n"
		"scores: %u \nmaintenance: %s \nstretch: %s \nimageless: %s",
		device_properties.deviceName,
		vsr_debug_device_type_print(device_properties.deviceType), device_properties.deviceID,
		scores, vsr_debug_maintainability_print(capabilities_device.has_swapchain_maintenance_1),
		vsr_debug_maintainability_print(capabilities_device.has_present_scaling_stretch),
		vsr_debug_maintainability_print(capabilities_device.has_imageless_frame_buffer)
	);
}

static inline const char * vsr_debug_device_type_print(const VkPhysicalDeviceType device_type) {
	switch( device_type ) {
	case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
		return "integrated";
	case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
		return "discrete";
	case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
		return "virtual";
	case VK_PHYSICAL_DEVICE_TYPE_OTHER:
		return "other";
	case VK_PHYSICAL_DEVICE_TYPE_CPU:
		return "central processing unit";
	default:
		return "unknown";
	}
}

static inline const char * vsr_debug_maintainability_print(bool flag) {
	return (flag == true) ? "supported" : "unsupported";
}
#endif

static bool vsr_window_initialize( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	glfwSetErrorCallback( vsr_callback_glfw_error );

	if(	glfwInit() != GLFW_TRUE ) {
		woem_push( "(vsr_window_initialize) GLFW initialization failed" );
		return false;
	}

	glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );

	application_pointer->window_pointer = glfwCreateWindow(
		VSR_WINDOW_WIDTH, VSR_WINDOW_HEIGHT, "VULKAN SQUARE ROTATION", NULL, NULL
	);
	if(	application_pointer->window_pointer == NULL ) {
		woem_push( "(vsr_window_initialize) window creation failed" );
		glfwTerminate();
		return false;
	}

	int frame_buffer_width, frame_buffer_height;
	glfwGetFramebufferSize(
		application_pointer->window_pointer, &frame_buffer_width, &frame_buffer_height
	);
	application_pointer->frame_state.width = frame_buffer_width;
	application_pointer->frame_state.height= frame_buffer_height;

	glfwSetWindowUserPointer(
		application_pointer->window_pointer, application_pointer
	);
	glfwSetKeyCallback(
		application_pointer->window_pointer, vsr_callback_glfw_key
	);
	glfwSetWindowIconifyCallback(
		application_pointer->window_pointer, vsr_callback_glfw_window_iconify
	);
	glfwSetFramebufferSizeCallback(
		application_pointer->window_pointer, vsr_callback_glfw_frame_buffer_size
	);

	application_pointer->is_initialized_glfw = true;

	return true;
}

static void vsr_callback_glfw_window_iconify( GLFWwindow * window_pointer, int iconified ) {
	struct VSR_Application * application_pointer = glfwGetWindowUserPointer( window_pointer );

	pthread_mutex_lock( &application_pointer->render_mutex );

	application_pointer->is_minimized = (iconified == GLFW_TRUE);

	pthread_cond_signal( &application_pointer->render_condition );
	pthread_mutex_unlock( &application_pointer->render_mutex );
}

static void vsr_callback_glfw_key(
		GLFWwindow * window_pointer, int key, int scancode, int action, int mods
	)
{
	(void) mods; (void) scancode;
	if(	action == GLFW_PRESS && key == GLFW_KEY_ESCAPE )
		glfwSetWindowShouldClose( window_pointer, GLFW_TRUE );
}

static void vsr_callback_glfw_frame_buffer_size(
		GLFWwindow * window_pointer, int width, int height
	)
{
	struct VSR_Application * application_pointer = glfwGetWindowUserPointer( window_pointer );

	pthread_mutex_lock( &application_pointer->render_mutex );

	application_pointer->frame_state.width	= width;
	application_pointer->frame_state.height	= height;
	if(	width > 0 && height > 0 ) {
		application_pointer->is_minimized					= false;
		application_pointer->frame_state.is_projection_dirty= true;
		application_pointer->frame_state.is_resize_pending	= true;
		application_pointer->last_resize_time_seconds		= glfwGetTime();
		uint_least64_t extent_packed = atomic_load_explicit(
			&application_pointer->swap_chain_extent_packed, memory_order_relaxed
		);
		uint32_t extent_width = (uint32_t)(extent_packed >> 32);
		uint32_t extent_height= (uint32_t)(extent_packed & UINT32_MAX);
		if(	(uint32_t) width > extent_width * VSR_RESIZE_INVALID_FACTOR ||
			(uint32_t) height> extent_height* VSR_RESIZE_INVALID_FACTOR )
			atomic_store_explicit(
				&application_pointer->is_swap_chain_valid, false, memory_order_relaxed
			);
	}
	else application_pointer->is_minimized = true;

	pthread_cond_signal( &application_pointer->render_condition );
	pthread_mutex_unlock( &application_pointer->render_mutex );
}

static void vsr_callback_glfw_error(int error_code, const char * description_pointer) {
	VSR_DEBUG_LOGF("(vsr_callback_glfw_error) error (%d): %s", error_code, description_pointer);
}
