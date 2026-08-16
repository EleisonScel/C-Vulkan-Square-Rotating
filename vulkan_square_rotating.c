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
#include "common/aligned_memory.h"
#include "common/cleanup_register.h"
#include "common/safe_multiplication.h"
#include "common/write_out_error_message.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>	/* glfwInit	*/

#include <cglm/cglm.h>	/* glm_rad	*/

#include <pthread.h>	/* pthread_		*/
#include <inttypes.h>	/* PRIu64		*/

#include <stdio.h>		/* fprintf		*/
#include <string.h>		/* strcmp		*/
#include <stdlib.h>		/* EXIT_STATUS	*/
#include <stdint.h>		/* uint32_t		*/
#include <stdbool.h>	/* bool			*/
#include <stdalign.h>	/* alignas()	*/

#ifndef NDEBUG
#	define VSR_DEBUG_LOG(format)		fprintf( stderr, (format"\n") )
#	define VSR_DEBUG_LOGF(format, ...)	fprintf( stderr, (format"\n"), __VA_ARGS__ )
#else
#	define VSR_DEBUG_LOG(format)		((void) 0)
#	define VSR_DEBUG_LOGF(format, ...)	((void) 0)
#endif

#define VSR_WINDOW_WWIDTH						800
#define VSR_WINDOW_HEIGHT						600

#define VSR_LIMIT_FRAMES_IN_FLIGHT				2
#define VSR_LIMIT_SWAPCHAIN_RECREATE_FAILURES	16
#define VSR_LIMIT_STACK_FAMILIES				64
#define VSR_LIMIT_DELETION_QUEUE_SIZE			256

struct VSR_Queue_Family_Indices {
	uint32_t graphics_family;
	uint32_t present_family;
	uint32_t transfer_family;
	bool has_graphics_family;
	bool has_present_family;
	bool has_transfer_family;
};

struct VSR_Deletion_Entity {
	alignas(64) VkImageView		* swap_chain_image_views_pointer;
	alignas(64) size_t			image_views_amount;
	alignas(64) VkFramebuffer	swap_chain_frame_buffer;
	alignas(64) VkSwapchainKHR	swap_chain;
	alignas(64) uint32_t		delete_frame;
};

struct VSR_Swap_Chain_Support_Details {
	struct VkSurfaceCapabilitiesKHR	surface_capabilities;
	uint32_t						formats_amount;
	struct VkSurfaceFormatKHR		* surface_formats_pointer;
	uint32_t						present_modes_amount;
	VkPresentModeKHR				* present_modes_pointer;
};

struct VSR_Synchronization_Objects {
	VkSemaphore	image_available_semaphore;
	VkSemaphore	render_finished_semaphore;
	VkFence		in_flight_fence;
	VkFence		present_fence;
};

struct VSR_Application {
	/* Hot data */
	VkDevice							device;
	VkQueue								graphics_queue;	/* implicitly destroyed with VkDevice */
	VkQueue								present_queue;	/* implicitly destroyed with VkDevice */
	VkSwapchainKHR						swap_chain;
	VkFramebuffer						swap_chain_frame_buffer;
	VkRenderPass						render_pass;
	VkPipelineLayout					pipeline_layout;
	VkPipeline							graphics_pipeline;
	VkBuffer							buffer_vertex;
	VkBuffer							buffer_index;

	/* backed on frames_in_flight_limit */
	VkBuffer							* buffers_uniform_pointer;
	VkDeviceMemory						buffers_uniform_memory;
	void								* buffers_uniform_mapped_pointer;

	/* auto freed with it's command pool destroying */
	VkDescriptorSet						* descriptor_sets_pointer;
	VkCommandBuffer						* command_buffers_pointer;
	struct VSR_Synchronization_Objects	* synchronization_objects_pointer;

	VkDeviceSize						buffer_uniform_alignment_size;
	VkDeviceSize						buffer_uniform_size;
	mat4								cached_view, cached_projection, cached_projection_view;

	float								spin_angle_current, spin_angle_rotation;
	uint32_t							swap_chain_image_views_amount;
	struct VkExtent2D					swap_chain_extent;
	VkFormat							swap_chain_image_format;

	/* for lower latency, otherwise back on swap_chain_images_pointer.size */
	uint8_t								frames_in_flight_limit;
	uint8_t								current_frame;
	bool								is_swap_chain_valid;
	bool								is_projection_dirty;
	bool								is_minimized;
	bool								is_running;

	/* Warm data */
	VkImageView							* swap_chain_image_views_pointer;/*image count backed*/
	VkImage								* swap_chain_images_pointer;
	int									width, height;
	struct RB_Ring_Buffer				deletion_queue;
	struct VSR_Deletion_Entity			deletion_entities[VSR_LIMIT_DELETION_QUEUE_SIZE];
	uint32_t							frame_counter;
	uint32_t							swap_chain_recreate_failed_amount;
	pthread_t							render_thread;
	pthread_mutex_t						render_mutex;
	pthread_cond_t						render_condition;

	/* Cold data */
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
	VkDescriptorPool					descriptor_pool;
	VkDescriptorSetLayout				descriptor_set_layout;
	struct VSR_Queue_Family_Indices		queue_family_indices;
	bool								is_initialized_glfw;
	bool								is_initialization_finished;
#ifndef NDEBUG
	VkDebugUtilsMessengerEXT			debug_messenger_function;
#endif
};

struct VSR_Vertex {
	vec2 position;
	vec3 color;
};

struct VSR_Uniform_Buffer_Object {
	mat4 model;
	mat4 view_projection;
};

static const struct VSR_Vertex vertices_array[] = {
	{ {-0.5f, -0.5f}, { 1.0f, 0.0f, 0.0f} },
	{ { 0.5f, -0.5f}, { 0.0f, 1.0f, 0.0f} },
	{ { 0.5f,  0.5f}, { 0.0f, 0.0f, 1.0f} },
	{ {-0.5f,  0.5f}, { 1.0f, 1.0f, 1.0f} }
};

static const uint16_t indices_array[] = {
	0, 1, 2, 2, 3, 0
};

static const char * instance_extensions_array[] = {
	VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
	VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
	VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME
};
static const uint32_t instance_extensions_amount =
	sizeof(instance_extensions_array) / sizeof(instance_extensions_array[0]);

static const char * device_extensions_array[] = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_KHR_IMAGELESS_FRAMEBUFFER_EXTENSION_NAME,	/* dependencies */
	VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,		/* 1			*/
	VK_KHR_MAINTENANCE2_EXTENSION_NAME,				/* 2			*/
	VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME	/* dependencies */
};
static const uint32_t device_extensions_amount =
	sizeof(device_extensions_array) / sizeof(device_extensions_array[0]);

#ifndef NDEBUG
static const char * validation_layers_array[] = {
	"VK_LAYER_KHRONOS_validation"
};
static const uint32_t validation_layers_amount =
	sizeof(validation_layers_array) / sizeof(validation_layers_array[0]);
#endif /* NDEBUG */

/* common */
static void vsr_application_initialization( struct VSR_Application * restrict application_pointer );

/* window scpecific */
static void vsr_key_callback( GLFWwindow * window_pointer, int key, int scancode, int action, int mods );
static void vsr_window_iconify_callback( GLFWwindow * window_pointer, int iconidied );
static void vsr_frame_buffer_size_callback( GLFWwindow * window_pointer, int width, int height );
static bool vsr_window_initialization( struct VSR_Application * restrict application_pointer );

/* Vulkan Initialization */
static bool vsr_create_surface( struct VSR_Application * restrict application_pointer );
static bool vsr_create_instance( struct VSR_Application * restrict application_pointer );
static bool vsr_vulkan_initialization( struct VSR_Application * restrict application_pointer );

/* Vulkan Debug messenger */
#ifndef NDEBUG
static void vsr_destroy_debug_utils_messenger_extension( VkInstance instance, VkDebugUtilsMessengerEXT debug_messenger, const VkAllocationCallbacks * restrict allocator_pointer );
static void vsr_populate_debug_messenger_create_information( struct VkDebugUtilsMessengerCreateInfoEXT * restrict creation_information_pointer );
static bool vsr_setup_debug_messenger( struct VSR_Application * restrict application_pointer );
static bool vsr_check_validation_layer_support(void);
static VkResult vsr_create_debug_utils_messenger_extension( VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT * restrict create_information_pointer, const VkAllocationCallbacks * restrict allocator_pointer, VkDebugUtilsMessengerEXT * restrict debug_messenger_pointer );
static VKAPI_ATTR VkBool32 VKAPI_CALL vsr_debug_callback_function( VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, VkDebugUtilsMessageTypeFlagsEXT message_type, const VkDebugUtilsMessengerCallbackDataEXT * restrict data_callback_pointer, void * restrict data_user_pointer );
#endif

/* extensions lists */
static const char ** vsr_get_required_extensions( uint32_t * restrict glfw_extension_amount_pointer );
static struct VkExtensionProperties * vsr_get_available_extensions( uint32_t * restrict extension_amount );

/* pick up GPUs */
static bool vsr_find_queue_families( VkSurfaceKHR surface, VkPhysicalDevice device, struct VSR_Queue_Family_Indices * restrict out_queue_family_indices_pointer );
static bool vsr_pick_physical_device( struct VSR_Application * restrict application_pointer );
static uint32_t vsr_rate_device_suitability( VkSurfaceKHR surface, VkPhysicalDevice device );
static bool vsr_check_device_extensions_support( VkPhysicalDevice device );
static bool vsr_queue_family_indices_is_complete( struct VSR_Queue_Family_Indices * restrict queue_family_indices_pointer );
/* device */
static bool vsr_create_logical_device( struct VSR_Application * restrict application_pointer, const char * restrict * restrict out_error_message_pointer );
static bool vsr_create_device_resources( struct VSR_Application * restrict application_pointer, const char * restrict * restrict out_error_message_pointer );
static bool vsr_device_recreate( struct VSR_Application * restrict application_pointer );
/* choose a swap chain part */
static void vsr_free_swap_chain_support_details( struct VSR_Swap_Chain_Support_Details * restrict swap_chain_support_pointer );
static bool vsr_create_swap_chain( struct VSR_Application * restrict application_pointer, VkSwapchainKHR * restrict out_swap_chain_pointer, VkImage ** out_swap_chain_images_pointer, uint32_t * restrict out_swap_chain_image_views_amount_pointer, VkFormat * restrict out_swap_chain_image_format_pointer, VkExtent2D * restrict out_swap_chain_extent_pointer, const char * restrict * restrict out_error_message_pointer );
static bool vsr_query_swap_chain_support( VkSurfaceKHR, VkPhysicalDevice, struct VSR_Swap_Chain_Support_Details * restrict out_swap_chain_support_details_pointer );
static struct VkExtent2D vsr_choose_swap_extent( const struct VkSurfaceCapabilitiesKHR * restrict surface_capabilities_pointer, GLFWwindow * restrict window_pointer );
static struct VkSurfaceFormatKHR vsr_choose_swap_surface_format( const struct VkSurfaceFormatKHR * restrict available_formats_pointer, size_t available_formats_amount );

/* swap chain images */
static bool vsr_create_image_views( struct VSR_Application * restrict application_pointer, VkImage * restrict swap_chain_images_pointer, uint32_t swap_chain_image_views_amount, VkFormat swap_chain_image_format, VkImageView ** out_swap_chain_image_views_pointer, const char * restrict * restrict out_error_message_pointer );
/* swap chain recreation */
static void vsr_recreate_swap_chain( struct VSR_Application * restrict application_pointer );
/* render */
static bool vsr_create_render_pass( struct VSR_Application * restrict application_pointer );
/* pipeline */
static bool vsr_create_graphics_pipeline( struct VSR_Application * restrict application_pointer );
static bool vsr_create_graphics_pipeline_from_shaders( struct VSR_Application * restrict application_pointer, const VkShaderModule shader_module_vertex, const VkShaderModule shader_module_fragment );
/* describe descriptor of bindings in shaders */
static bool vsr_create_descriptor_set_layout( struct VSR_Application * restrict application_pointer );
/* descriptor */
static bool vsr_create_descriptor_pool( struct VSR_Application * restrict application_pointer );
/* descriptor set */
static bool vsr_create_descriptor_sets( struct VSR_Application * restrict application_pointer );
/* shaders */
static bool vsr_create_shader_module( VkDevice device, char * restrict shader_code_source_pointer, size_t file_size, VkShaderModule * restrict out_shader_module_pointer );
/* frame_buffers */
static bool vsr_create_frame_buffer( struct VSR_Application * restrict application_pointer, VkFormat swap_chain_image_format, VkExtent2D swap_chain_extent, VkFramebuffer * restrict out_swap_chain_frame_buffer_pointer, const char * restrict * restrict out_error_message_pointer );
/* commands handle */
static bool vsr_create_command_pools( struct VSR_Application * restrict application_pointer );
static bool vsr_record_command_buffer( struct VSR_Application * restrict application_pointer, VkCommandBuffer command_buffer, uint32_t image_index );
static bool vsr_create_command_buffers( struct VSR_Application * restrict application_pointer );
static bool vsr_create_inclusive_command_pool( struct VSR_Application * restrict application_pointer, VkCommandPool * restrict command_pool_pointer, VkCommandPoolCreateFlags flags, uint32_t family );
/* buffers */
static inline bool vsr_create_staging_buffer( struct VSR_Application * restrict application_pointer, VkDeviceSize buffer_size, VkBuffer * restrict out_buffer_pointer, VkDeviceMemory * out_buffer_memory, const char * restrict function_name_pointer );
static bool vsr_copy_buffer( struct VSR_Application * restrict application_pointer, VkBuffer buffer_source, VkBuffer buffer_destination, VkDeviceSize size, VkCommandPool command_pool );
static bool vsr_create_buffer( struct VSR_Application * restrict application_pointer, VkDeviceSize size, VkBufferUsageFlags usage, const VkMemoryPropertyFlags * restrict properties_lists[static 2], int properties_required_amount, VkBuffer * restrict out_buffer_pointer, VkDeviceMemory * out_buffer_memory_pointer, const char * restrict function_name_pointer );
static bool vsr_create_fast_buffers( struct VSR_Application * restrict application_pointer );
static bool vsr_create_uniforms_buffer( struct VSR_Application * restrict application_pointer );
static bool vsr_create_inclusive_buffer( struct VSR_Application * restrict application_pointer, void const * buffer, VkDeviceSize bufferSize, VkBufferUsageFlags buffer_usage, VkBuffer staging_buffer, VkDeviceMemory staging_buffer_memory, VkCommandPool command_pool, VkBuffer * restrict out_buffer_pointer, VkDeviceMemory * out_buffer_memory_pointer, const char * restrict function_name_pointer );
/* update uniform buffer data */
static void vsr_update_buffer_uniform( struct VSR_Application * restrict application_pointer, uint32_t current_image );
/* binding buffer description */
static struct VkVertexInputBindingDescription vsr_get_binding_description(void);
/* vertex attribute description */
static struct VkVertexInputAttributeDescription * vsr_get_attribute_descriptions( uint32_t * restrict out_size_pointer );
/* find memory type on GPU */
static bool vsr_find_memory_type( struct VSR_Application * restrict application_pointer, uint32_t type_filter, const VkMemoryPropertyFlags * restrict properties_list_required, const VkMemoryPropertyFlags * restrict properties_list_forbidden, int properties_required_amount, uint32_t * restrict out_type_index_pointer );
/* synchronization handle */
static bool vsr_synchronization_objects_create( struct VSR_Application * restrict application_pointer );
/* delayed deletion */
static void vsr_process_delay_deletion( struct VSR_Application * restrict application_pointer );
static bool vsr_deletion_cleanup_swap_chain( struct VSR_Application * restrict application_pointer, VkSwapchainKHR swap_chain, VkImageView * restrict swap_chain_image_views_pointer, size_t image_views_amount, VkFramebuffer swap_chain_frame_buffer, const char * restrict * restrict out_error_message_pointer );

/* draw frame */
static void vsr_draw_frame( struct VSR_Application * restrict application_pointer );

/* app main cycle */
static void vsr_main_loop( struct VSR_Application * restrict application_pointer );
static void * vsr_render_thread_function( void * restrict argument_pointer );

/* app clean up */
static void vsr_cleanup_swap_chain( struct VSR_Application * restrict application_pointer );
static void vsr_cleanup_application( void * restrict argument_pointer );
static void vsr_destroy_deletion_entity( struct VSR_Application * restrict application_pointer, struct VSR_Deletion_Entity * restrict entity_pointer );
static void vsr_destroy_device_and_resources( struct VSR_Application * restrict application_pointer );
static void vsr_destroy_synchronization_objects( struct VSR_Application * restrict application_pointer, uint8_t objects_amount );

int main (void) {
	struct VSR_Application application = { 0 };
	bool initialized_application = false;

	cr_register_cleanup_wrapper( vsr_cleanup_application, &application );
	vsr_application_initialization( &application );

	pthread_mutex_init( &application.render_mutex, NULL );
	pthread_cond_init( &application.render_condition, NULL );

	if( pthread_create(
			&application.render_thread, NULL, vsr_render_thread_function,
			&application
		) != 0 )
	{
		woem_push( "(vsr_main_loop) render thread creation failed" );
	}
	else {
		initialized_application =
			vsr_window_initialization( &application ) == true &&
			vsr_vulkan_initialization( &application ) == true;

		pthread_mutex_lock( &application.render_mutex );
		application.is_running = initialized_application;
		application.is_initialization_finished = true;
		pthread_cond_signal( &application.render_condition );
		pthread_mutex_unlock( &application.render_mutex );

		if( initialized_application == true )
		{
			vsr_main_loop( &application );

			pthread_mutex_lock( &application.render_mutex );
			application.is_running = false;
			pthread_cond_signal( &application.render_condition );
			pthread_mutex_unlock( &application.render_mutex );
		}

		pthread_join( application.render_thread, NULL );
		if ( application.device != VK_NULL_HANDLE )
			vkDeviceWaitIdle( application.device );
	}

	pthread_cond_destroy( &application.render_condition );
	pthread_mutex_destroy( &application.render_mutex );

	bool message_have_to_be_freed;
	for ( char * error_message_pointer;
			(error_message_pointer = woem_pop(&message_have_to_be_freed)) != NULL; )
	{
		fprintf( stderr, "error: %s\n", error_message_pointer );
		if( message_have_to_be_freed == true )
			free( error_message_pointer );
	}

	exit( initialized_application == true ? EXIT_SUCCESS : EXIT_FAILURE );
}

static void * vsr_render_thread_function( void * restrict argument_pointer ) {
	struct VSR_Application * application_pointer = (struct VSR_Application *) argument_pointer;
	assert_m( application_pointer != NULL, "No application found" );

	pthread_mutex_lock( &application_pointer->render_mutex );

	while ( application_pointer->is_initialization_finished == false )
		pthread_cond_wait(
			&application_pointer->render_condition, &application_pointer->render_mutex
		);

	while ( application_pointer->is_running == true ) {
		if( application_pointer->is_minimized == true ) {
			pthread_cond_wait(
				&application_pointer->render_condition, &application_pointer->render_mutex
			);
			continue;
		}

		pthread_mutex_unlock(&application_pointer->render_mutex );
		vsr_draw_frame( application_pointer );
		pthread_mutex_lock( &application_pointer->render_mutex );
	}

	pthread_mutex_unlock( &application_pointer->render_mutex );

	return NULL;
}

static bool vsr_vulkan_initialization( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );
	const char * error_message_pointer;

	if ( vsr_create_instance(					application_pointer ) == false ) return false;

#ifndef NDEBUG
	if ( vsr_setup_debug_messenger(				application_pointer ) == false ) return false;
#endif

	if ( vsr_create_surface(					application_pointer ) == false ) return false;
	if ( vsr_pick_physical_device(				application_pointer ) == false ) return false;
	if( vsr_create_logical_device(	application_pointer, &error_message_pointer ) == false ||
		vsr_create_device_resources(application_pointer, &error_message_pointer ) == false )
	{
		woem_push( "%s", error_message_pointer );
		return false;
	}

	return true;
}

static bool vsr_create_device_resources(
		struct VSR_Application * restrict application_pointer,
		const char * restrict * restrict out_error_message_pointer
	)
{
	assert_m( application_pointer		!= NULL, "No application found"				);
	assert_m( out_error_message_pointer	!= NULL, "No error message storage found"	);

	if ( vsr_create_swap_chain(
			application_pointer, &application_pointer->swap_chain,
			&application_pointer->swap_chain_images_pointer,
			&application_pointer->swap_chain_image_views_amount,
			&application_pointer->swap_chain_image_format,
			&application_pointer->swap_chain_extent, out_error_message_pointer
		) == false )
		return false;

	if ( rb_ring_buffer_initialize_static(
			&application_pointer->deletion_queue, application_pointer->deletion_entities,
			VSR_LIMIT_DELETION_QUEUE_SIZE, sizeof(struct VSR_Deletion_Entity), 0, 0, 0
		) == false )
		return false;
	if ( vsr_create_image_views(
			application_pointer, application_pointer->swap_chain_images_pointer,
			application_pointer->swap_chain_image_views_amount,
			application_pointer->swap_chain_image_format,
			&application_pointer->swap_chain_image_views_pointer, out_error_message_pointer
		) == false )
		return false;

	if ( vsr_create_render_pass(				application_pointer ) == false ) return false;
	if ( vsr_create_descriptor_set_layout(		application_pointer ) == false ) return false;
	if ( vsr_create_graphics_pipeline(			application_pointer ) == false ) return false;

	if ( vsr_create_frame_buffer(
			application_pointer, application_pointer->swap_chain_image_format,
			application_pointer->swap_chain_extent,
			&application_pointer->swap_chain_frame_buffer, out_error_message_pointer
		) == false )
		return false;

	if ( vsr_create_command_pools(				application_pointer ) == false ) return false;
	if ( vsr_create_fast_buffers(				application_pointer ) == false ) return false;
	if ( vsr_create_uniforms_buffer(			application_pointer ) == false ) return false;
	if ( vsr_create_descriptor_pool(			application_pointer ) == false ) return false;
	if ( vsr_create_descriptor_sets(			application_pointer ) == false ) return false;
	if ( vsr_create_command_buffers(			application_pointer ) == false ) return false;
	if ( vsr_synchronization_objects_create(	application_pointer ) == false ) return false;

	application_pointer->is_swap_chain_valid = true;

	return true;
}

static void vsr_main_loop( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	while ( glfwWindowShouldClose( application_pointer->window_pointer ) == false )
		glfwWaitEvents();
}

static void vsr_application_initialization(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	*application_pointer = (struct VSR_Application) {
		.width					= VSR_WINDOW_WWIDTH,
		.height					= VSR_WINDOW_HEIGHT,
		/* 2 or must be backed by swap_chain swap_chain_image_views_amount */
		.frames_in_flight_limit	= VSR_LIMIT_FRAMES_IN_FLIGHT,
		.spin_angle_rotation	= 0.02f
	};
}

static bool vsr_device_recreate( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	const char * error_message_pointer = NULL;

	vsr_destroy_device_and_resources( application_pointer );

	if( vsr_create_logical_device( application_pointer, &error_message_pointer )== false ||
		vsr_create_device_resources(application_pointer, &error_message_pointer)== false )
	{
		VSR_DEBUG_LOGF( "%s", error_message_pointer );
		return false;
	}

	return true;
}

static void vsr_destroy_device_and_resources(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	if ( application_pointer->device == VK_NULL_HANDLE )
		return;

	vkDeviceWaitIdle( application_pointer->device );

	if ( application_pointer->command_buffers_pointer != NULL ) {
		free( application_pointer->command_buffers_pointer );
		application_pointer->command_buffers_pointer = NULL;
	}

	struct VSR_Deletion_Entity * entity_pointer;
	while ((entity_pointer = rb_ring_buffer_pop(&application_pointer->deletion_queue)) != NULL)
		vsr_destroy_deletion_entity( application_pointer, entity_pointer );

	vsr_cleanup_swap_chain( application_pointer );

	if ( application_pointer->buffers_uniform_pointer != NULL ) {
		for ( uint8_t buffer_uniform_index = 0;
				buffer_uniform_index < application_pointer->frames_in_flight_limit;
				++buffer_uniform_index )
		{
			if(application_pointer->buffers_uniform_pointer[buffer_uniform_index]
				!= VK_NULL_HANDLE)
				vkDestroyBuffer(
					application_pointer->device,
					application_pointer->buffers_uniform_pointer[buffer_uniform_index], NULL
				);
		}
		free( application_pointer->buffers_uniform_pointer );
		application_pointer->buffers_uniform_pointer = NULL;
	}

	if ( application_pointer->buffers_uniform_memory != VK_NULL_HANDLE ) {
		if ( application_pointer->buffers_uniform_mapped_pointer != NULL ) {
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

	if ( application_pointer->descriptor_pool != VK_NULL_HANDLE )
		vkDestroyDescriptorPool(
			application_pointer->device, application_pointer->descriptor_pool, NULL
		);

	if ( application_pointer->descriptor_sets_pointer != NULL ) {
		free( application_pointer->descriptor_sets_pointer );
		application_pointer->descriptor_sets_pointer = NULL;
	}

	if ( application_pointer->descriptor_set_layout != VK_NULL_HANDLE )
		vkDestroyDescriptorSetLayout(
			application_pointer->device, application_pointer->descriptor_set_layout, NULL
		);

	if ( application_pointer->buffer_vertex != VK_NULL_HANDLE )
		vkDestroyBuffer(application_pointer->device, application_pointer->buffer_vertex, NULL);
	if ( application_pointer->buffer_memory_vertex != VK_NULL_HANDLE )
		vkFreeMemory(
			application_pointer->device, application_pointer->buffer_memory_vertex, NULL
		);

	if ( application_pointer->buffer_index != VK_NULL_HANDLE )
		vkDestroyBuffer( application_pointer->device, application_pointer->buffer_index, NULL );
	if ( application_pointer->buffer_memory_index != VK_NULL_HANDLE )
		vkFreeMemory(
			application_pointer->device, application_pointer->buffer_memory_index, NULL
		);

	if ( application_pointer->graphics_pipeline != VK_NULL_HANDLE )
		vkDestroyPipeline(
			application_pointer->device, application_pointer->graphics_pipeline, NULL
		);
	if ( application_pointer->pipeline_layout != VK_NULL_HANDLE )
		vkDestroyPipelineLayout(
			application_pointer->device, application_pointer->pipeline_layout, NULL
		);
	if ( application_pointer->render_pass != VK_NULL_HANDLE )
		vkDestroyRenderPass(
			application_pointer->device, application_pointer->render_pass, NULL
		);

	vsr_destroy_synchronization_objects(
		application_pointer, application_pointer->frames_in_flight_limit
	);
	if ( application_pointer->synchronization_objects_pointer != NULL ) {
		free( application_pointer->synchronization_objects_pointer );
		application_pointer->synchronization_objects_pointer = NULL;
	}

	if ( application_pointer->command_pool_graphic != VK_NULL_HANDLE )
		vkDestroyCommandPool(
			application_pointer->device, application_pointer->command_pool_graphic, NULL
		);
	if ( application_pointer->command_pool_transfer != VK_NULL_HANDLE )
		vkDestroyCommandPool(
			application_pointer->device, application_pointer->command_pool_transfer, NULL
		);

	vkDestroyDevice( application_pointer->device, NULL );
	application_pointer->device = VK_NULL_HANDLE; 
}

static void vsr_cleanup_swap_chain( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	if ( application_pointer->swap_chain_frame_buffer != VK_NULL_HANDLE ) {
		vkDestroyFramebuffer(
			application_pointer->device, application_pointer->swap_chain_frame_buffer, NULL
		);
		application_pointer->swap_chain_frame_buffer = NULL;
	}

	if ( application_pointer->swap_chain_image_views_pointer != NULL ) {
		for (size_t image = 0;
				image < application_pointer->swap_chain_image_views_amount; ++image)
			vkDestroyImageView(
				application_pointer->device,
				application_pointer->swap_chain_image_views_pointer[image], NULL
			);
		free( application_pointer->swap_chain_image_views_pointer );
		application_pointer->swap_chain_image_views_pointer = NULL;
	}

	if ( application_pointer->swap_chain != VK_NULL_HANDLE ) {
		vkDestroySwapchainKHR(
			application_pointer->device, application_pointer->swap_chain, NULL
		);
		application_pointer->swap_chain = VK_NULL_HANDLE;
	}

	if ( application_pointer->swap_chain_images_pointer != NULL ) {
		free( application_pointer->swap_chain_images_pointer );
		application_pointer->swap_chain_images_pointer = NULL;
	}
}

static void vsr_cleanup_application( void * restrict argument_pointer ) {
	if (argument_pointer == NULL)
		return;

	struct VSR_Application * application_pointer = (struct VSR_Application *) argument_pointer;
	assert_m( application_pointer != NULL, "No application found" );

	vsr_destroy_device_and_resources( application_pointer );

	if ( application_pointer->instance != VK_NULL_HANDLE ) {

#ifndef NDEBUG
		if ( application_pointer->debug_messenger_function != VK_NULL_HANDLE )
			vsr_destroy_debug_utils_messenger_extension(
				application_pointer->instance, application_pointer->debug_messenger_function,
				NULL
			);
#endif

		if ( application_pointer->surface != VK_NULL_HANDLE ) {
			vkDestroySurfaceKHR(
				application_pointer->instance, application_pointer->surface, NULL
			);
			application_pointer->surface = NULL;
		}
		vkDestroyInstance( application_pointer->instance, NULL );
		application_pointer->instance = NULL;
	}
	if ( application_pointer->window_pointer != NULL ) {
		glfwDestroyWindow( application_pointer->window_pointer );
		application_pointer->window_pointer = NULL;
	}
	if ( application_pointer->is_initialized_glfw == true )
		glfwTerminate();

	application_pointer = NULL;
}

static bool vsr_deletion_cleanup_swap_chain(
		struct VSR_Application * restrict application_pointer, VkSwapchainKHR swap_chain,
		VkImageView * restrict swap_chain_image_views_pointer, size_t image_views_amount,
		VkFramebuffer swap_chain_frame_buffer,
		const char * restrict * restrict out_error_message_pointer
	)
{
	assert_m( application_pointer		!= NULL, "No application found"				);
	assert_m( out_error_message_pointer	!= NULL, "No error message storage found"	);

	if ( rb_ring_buffer_is_full( &application_pointer->deletion_queue ) == true ) {
		vkDeviceWaitIdle( application_pointer->device );

		while ( rb_ring_buffer_peek(&application_pointer->deletion_queue) != NULL ) {
			struct VSR_Deletion_Entity * entity_pointer = rb_ring_buffer_pop(
				&application_pointer->deletion_queue
			);
			vsr_destroy_deletion_entity( application_pointer, entity_pointer );
		}

		application_pointer->frame_counter = 0;
	}

	struct VSR_Deletion_Entity deletion = {
		.swap_chain_image_views_pointer	= swap_chain_image_views_pointer,
		.image_views_amount				= image_views_amount,
		.swap_chain_frame_buffer		= swap_chain_frame_buffer,
		.swap_chain						= swap_chain,
		.delete_frame					= application_pointer->frame_counter +
			application_pointer->frames_in_flight_limit
	};

	if( rb_ring_buffer_push( &application_pointer->deletion_queue, &deletion ) == false ) {
		*out_error_message_pointer =
			"(vsr_deletion_cleanup_swap_chain) failed to push to deletion queue";
		return false;
	}

	return true;
}

static void vsr_destroy_deletion_entity(
		struct VSR_Application * restrict application_pointer,
		struct VSR_Deletion_Entity * entity_pointer
	)
{
	assert_m( application_pointer	!= NULL, "No application found" );
	assert_m( entity_pointer		!= NULL, "No application found" );

	if ( entity_pointer->swap_chain_frame_buffer != VK_NULL_HANDLE )
		vkDestroyFramebuffer(
			application_pointer->device, entity_pointer->swap_chain_frame_buffer, NULL
		);
	if ( entity_pointer->swap_chain != VK_NULL_HANDLE )
		vkDestroySwapchainKHR(
			application_pointer->device, entity_pointer->swap_chain, NULL
		);
	if ( entity_pointer->swap_chain_image_views_pointer != NULL ) {
		for ( size_t image_view_index = 0;
				image_view_index < entity_pointer->image_views_amount; ++image_view_index )
		{
			if(entity_pointer->swap_chain_image_views_pointer[image_view_index]
				!= VK_NULL_HANDLE)
			{
				vkDestroyImageView(
					application_pointer->device,
					entity_pointer->swap_chain_image_views_pointer[image_view_index], NULL
				);
			}
		}
		free( entity_pointer->swap_chain_image_views_pointer );
	}
}

static void vsr_process_delay_deletion( struct VSR_Application * restrict application_pointer )
{
	assert_m( application_pointer != NULL, "No application found" );

	for ( size_t entity_index = application_pointer->deletion_queue.amount;
			entity_index > 0; --entity_index )
	{
		struct VSR_Deletion_Entity * entity_pointer = rb_ring_buffer_peek(
			&application_pointer->deletion_queue
		);
		if ( entity_pointer == NULL )
			break;

		if ( application_pointer->frame_counter < entity_pointer->delete_frame )
			break;

		vsr_destroy_deletion_entity( application_pointer, entity_pointer );
		rb_ring_buffer_discard( &application_pointer->deletion_queue );
	}

	if ( application_pointer->deletion_queue.amount == 0 )
		application_pointer->frame_counter = 0;
	else ++application_pointer->frame_counter;
}

static bool vsr_create_descriptor_sets( struct VSR_Application * restrict application_pointer )
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

	if( sa_malloc_array(
			&application_pointer->descriptor_sets_pointer,
			application_pointer->frames_in_flight_limit, sizeof(VkDescriptorSet)
		) == false )
	{
		ep_exit_print( "(vsr_create_descriptor_sets) allocation size overflow" );
	}
	else if ( application_pointer->descriptor_sets_pointer == NULL ) {
		woem_push( "(vsr_create_descriptor_sets) descriptor sets allocation failed" );
		return false;
	}

	if (vkAllocateDescriptorSets(
			application_pointer->device, &allocate_information,
			application_pointer->descriptor_sets_pointer
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_create_descriptor_sets) descriptor sets VRAM allocation failed" );
		return false;
	}

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

	return true;
}

static bool vsr_create_descriptor_pool( struct VSR_Application * restrict application_pointer )
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

	if( vkCreateDescriptorPool(
			application_pointer->device, &pool_information, NULL,
			&application_pointer->descriptor_pool
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_create_descriptor_pool) descriptor pool creation failed" );
		return false;
	}

	return true;
}

static bool vsr_create_descriptor_set_layout(
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

	if( vkCreateDescriptorSetLayout(
			application_pointer->device, &layout_create_information, NULL,
			&application_pointer->descriptor_set_layout
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_create_descriptor_set_layout) descriptor set layout creation failed" );
		return false;
	}

	return true;
}

static bool vsr_create_uniforms_buffer( struct VSR_Application * restrict application_pointer )
{
	assert_m( application_pointer != NULL, "No application found" );

	application_pointer->buffer_uniform_size = sizeof( struct VSR_Uniform_Buffer_Object );

	VkPhysicalDeviceProperties device_properties;
	vkGetPhysicalDeviceProperties( application_pointer->device_physical, &device_properties );

	VkDeviceSize safe_alignment_required =
		(device_properties.limits.minUniformBufferOffsetAlignment == 0)
		? 1
		: device_properties.limits.minUniformBufferOffsetAlignment;
	if (sa_ovf_round_up_uint64_t(
			application_pointer->buffer_uniform_size, safe_alignment_required,
			&application_pointer->buffer_uniform_alignment_size
		) == true)
	{
		woem_push( "(vsr_create_uniforms_buffer) uniform buffer alignment calculus failed" );
		return false;
	}

	if ( application_pointer->buffers_uniform_pointer != VK_NULL_HANDLE ) {
		for(uint8_t buffer_uniform_index = 0;
				buffer_uniform_index < application_pointer->frames_in_flight_limit;
			++buffer_uniform_index )
		{
			if ( application_pointer->buffers_uniform_pointer[buffer_uniform_index] != NULL )
				vkDestroyBuffer(
					application_pointer->device,
					application_pointer->buffers_uniform_pointer[buffer_uniform_index], NULL
				);
		}
		free( application_pointer->buffers_uniform_pointer );
		application_pointer->buffers_uniform_pointer = NULL;
	}

	if ( application_pointer->buffers_uniform_memory != VK_NULL_HANDLE ) {
		if ( application_pointer->buffers_uniform_mapped_pointer != NULL ) {
			vkUnmapMemory(
				application_pointer->device, application_pointer->buffers_uniform_memory
			);
			application_pointer->buffers_uniform_mapped_pointer = NULL;
		}
		vkFreeMemory(
			application_pointer->device, application_pointer->buffers_uniform_memory, NULL
		);
		application_pointer->buffers_uniform_memory = VK_NULL_HANDLE;
	}

	if( sa_malloc_array(
			&application_pointer->buffers_uniform_pointer,
			application_pointer->frames_in_flight_limit,
			sizeof(*application_pointer->buffers_uniform_pointer)
		) == false )
	{
		ep_exit_print( "(vsr_create_uniforms_buffer) allocation size overflow" );
	}
	else if ( application_pointer->buffers_uniform_pointer == NULL ) {
		woem_push( "(vsr_create_uniforms_buffer) allocation of uniform buffers failed" );
		return false;
	}

	struct VkBufferCreateInfo buffer_create_information = {
		.sType			= VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size			= application_pointer->buffer_uniform_alignment_size,
		.usage			= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode	= VK_SHARING_MODE_EXCLUSIVE
	};

	for(uint8_t buffer_uniform_index = 0;
			buffer_uniform_index < application_pointer->frames_in_flight_limit;
		++buffer_uniform_index )
	{
		if (vkCreateBuffer(
				application_pointer->device, &buffer_create_information, NULL,
				&application_pointer->buffers_uniform_pointer[buffer_uniform_index]
			) != VK_SUCCESS )
		{
			woem_push( "(vsr_create_uniforms_buffer) buffer creation failed" );
			return false;
		}
	}

	VkMemoryRequirements memory_requirements;
	vkGetBufferMemoryRequirements(
		application_pointer->device, application_pointer->buffers_uniform_pointer[0],
		&memory_requirements
	);

	const VkMemoryPropertyFlags properties_list_required[] = {
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
	};
	const VkMemoryPropertyFlags properties_list_forbidden[] = {
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		0
	};
	const int properties_lists_amount =
		sizeof(properties_list_required) / sizeof(*properties_list_required);

	uint32_t memory_type_index;
	if( vsr_find_memory_type(
			application_pointer, memory_requirements.memoryTypeBits, properties_list_required,
			properties_list_forbidden, properties_lists_amount, &memory_type_index
		) == false )
	{
		woem_push( "(vsr_create_uniforms_buffer) suitable memory type not found" );
		return false;
	}

	VkDeviceSize allocation_size;
	if( sa_ovf_mul_uint64_t(
			memory_requirements.size, application_pointer->frames_in_flight_limit,
			&allocation_size
		) == true )
	{
		woem_push( "(vsr_create_uniforms_buffer) invalid allocation size" );
		return false;
	}

	safe_alignment_required = (memory_requirements.alignment == 0)
		? 1
		: memory_requirements.alignment;
	if( sa_ovf_round_up_uint64_t(
			allocation_size, safe_alignment_required, &allocation_size
		) == true )
	{
		woem_push( "(vsr_create_uniforms_buffer) allocation size alignment overflow" );
		return false;
	}

	struct VkMemoryAllocateInfo allocate_information = {
		.sType				= VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize		= allocation_size,
		.memoryTypeIndex	= memory_type_index
	};

	if( vkAllocateMemory(
			application_pointer->device, &allocate_information, NULL,
			&application_pointer->buffers_uniform_memory
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_create_uniforms_buffer) vertex buffer memory allocation failed" );
		return false;
	}

	if( vkMapMemory(
			application_pointer->device, application_pointer->buffers_uniform_memory, 0,
			VK_WHOLE_SIZE, 0, &application_pointer->buffers_uniform_mapped_pointer
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_create_uniforms_buffer) memory mapping failed" );
		return false;
	}

	for ( uint8_t buffer_uniform_index = 0;
			buffer_uniform_index < application_pointer->frames_in_flight_limit;
			++buffer_uniform_index )
	{
		VkDeviceSize size_offset;
		if( sa_ovf_mul_uint64_t(
				buffer_uniform_index, memory_requirements.size, &size_offset
			) == true )
		{
			woem_push( "(vsr_create_uniforms_buffer) invalid offset size" );
			return false;
		}
		if( vkBindBufferMemory(
				application_pointer->device,
				application_pointer->buffers_uniform_pointer[buffer_uniform_index],
				application_pointer->buffers_uniform_memory, size_offset
			) != VK_SUCCESS )
		{
			woem_push( "(vsr_create_uniforms_buffer) binding buffer memory  failed" );
			return false;
		}
	}

	return true;
}

static bool vsr_create_fast_buffers( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	VkDeviceMemory	staging_buffer_memory	= VK_NULL_HANDLE;
	VkCommandPool	command_pool			= VK_NULL_HANDLE;
	VkBuffer		staging_buffer			= VK_NULL_HANDLE;

	const VkDeviceSize staging_union_size = sizeof(vertices_array) > sizeof(indices_array)
		? sizeof(vertices_array)
		: sizeof(indices_array);

	if ( vsr_create_staging_buffer(
			application_pointer, staging_union_size, &staging_buffer, &staging_buffer_memory,
			"vsr_create_fast_buffers"
		) == false )
		return false;

	if ( vsr_create_inclusive_command_pool(
			application_pointer, &command_pool, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			application_pointer->queue_family_indices.transfer_family
		) == false )
		goto cleanup;

	if( vsr_create_inclusive_buffer(
			application_pointer,
			vertices_array,
			sizeof(vertices_array),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			staging_buffer, staging_buffer_memory, command_pool,
			&application_pointer->buffer_vertex,
			&application_pointer->buffer_memory_vertex,
			"vsr_create_fast_buffers"
		) == false )
		goto cleanup_pool;

	if ( vsr_create_inclusive_buffer(
			application_pointer,
			indices_array,
			sizeof(indices_array),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			staging_buffer, staging_buffer_memory, command_pool,
			&application_pointer->buffer_index,
			&application_pointer->buffer_memory_index,
			"vsr_create_fast_buffers"
		) == false )
		goto cleanup_pool;

	vkDestroyCommandPool( application_pointer->device, command_pool, NULL );
	vkDestroyBuffer( application_pointer->device, staging_buffer, NULL );
	vkFreeMemory( application_pointer->device, staging_buffer_memory, NULL );

	return true;

cleanup_pool:
	vkDestroyCommandPool( application_pointer->device, command_pool, NULL );

cleanup:
	vkDestroyBuffer( application_pointer->device, staging_buffer, NULL );
	vkFreeMemory( application_pointer->device, staging_buffer_memory, NULL );
	return false;
}

static inline bool vsr_create_staging_buffer(
		struct VSR_Application * restrict application_pointer, VkDeviceSize buffer_size,
		VkBuffer * restrict out_buffer_pointer, VkDeviceMemory * out_buffer_memory_pointer,
		const char * restrict function_name_pointer
	)
{
	assert_m( application_pointer		!= NULL, "No application found"				);
	assert_m( out_buffer_pointer		!= NULL, "No buffer storage found"			);
	assert_m( out_buffer_memory_pointer	!= NULL, "No buffer memory storage found"	);
	assert_m( function_name_pointer		!= NULL, "No function name found"			);

	return vsr_create_buffer(
		application_pointer,
		buffer_size,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		(const VkMemoryPropertyFlags *[2]) {
			(const VkMemoryPropertyFlags [])
			{
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
			},
			(const VkMemoryPropertyFlags [])
			{
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				0
			}
		},
		2,
		out_buffer_pointer,
		out_buffer_memory_pointer,
		function_name_pointer
	);
}

static bool vsr_create_inclusive_buffer(
		struct VSR_Application * restrict application_pointer, void const * buffer,
		VkDeviceSize buffer_size, VkBufferUsageFlags buffer_usage, VkBuffer staging_buffer,
		VkDeviceMemory staging_buffer_memory, VkCommandPool command_pool,
		VkBuffer * restrict out_buffer_pointer, VkDeviceMemory * out_buffer_memory_pointer,
		const char * restrict function_name_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	void * data_pointer = NULL;
	if ( vkMapMemory(
			application_pointer->device, staging_buffer_memory, 0, VK_WHOLE_SIZE, 0,
			&data_pointer
		) != VK_SUCCESS )
	{
		woem_push( "(%s) staging buffer memory mapping failed", function_name_pointer );
		return false;
	}
	memcpy( data_pointer, buffer, (size_t) buffer_size );
	
	VkMappedMemoryRange flush_range = {
		.sType	= VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
		.memory	= staging_buffer_memory,
		.size	= VK_WHOLE_SIZE
	};
	if( vkFlushMappedMemoryRanges( application_pointer->device, 1, &flush_range ) != VK_SUCCESS)
	{
		woem_push( "(%s) staging buffer memory mapping failed", function_name_pointer );
		vkUnmapMemory( application_pointer->device, staging_buffer_memory );
		return false;
	}

	vkUnmapMemory( application_pointer->device, staging_buffer_memory );

	if ( vsr_create_buffer(
			application_pointer,
			buffer_size,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | buffer_usage,
			(const VkMemoryPropertyFlags *[2]) {
				(const VkMemoryPropertyFlags [])
				{
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					0
				},
				(const VkMemoryPropertyFlags [])
				{
					0,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
					0
				}
			},
			3,
			out_buffer_pointer,
			out_buffer_memory_pointer,
			"vsr_create_inclusive_buffer"
		) == false )
		return false;

	return vsr_copy_buffer(
		application_pointer, staging_buffer, *out_buffer_pointer, buffer_size, command_pool
	);
}

static bool vsr_copy_buffer(
		struct VSR_Application * restrict application_pointer, VkBuffer buffer_source,
		VkBuffer buffer_destination, VkDeviceSize size, VkCommandPool command_pool
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	bool value_to_return = false;

	struct VkCommandBufferAllocateInfo allocate_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool		= command_pool,
		.level				= VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount	= 1
	};

	VkCommandBuffer command_buffer;
	if (vkAllocateCommandBuffers(
			application_pointer->device, &allocate_information, &command_buffer
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_copy_buffer) command buffer allocation failed" );
		return false;
	}

	struct VkBufferCopy region_to_copy;
	struct VkSubmitInfo submit_information;
	struct VkCommandBufferBeginInfo command_buffer_begin_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		/* type of command buffer using */
		.flags				= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	if (vkBeginCommandBuffer( command_buffer, &command_buffer_begin_information )
		!= VK_SUCCESS )
	{
		woem_push( "(vsr_copy_buffer) recording command buffer beginning failed" );
		goto out;
	}

	region_to_copy = (struct VkBufferCopy) { .size = size };

	vkCmdCopyBuffer( command_buffer, buffer_source, buffer_destination, 1, &region_to_copy );

	if (vkEndCommandBuffer( command_buffer )
		!= VK_SUCCESS )
	{
		woem_push( "(vsr_copy_buffer) command buffer recording failed" );
		goto out;
	}

	submit_information = (struct VkSubmitInfo) {
		.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount	= 1,
		.pCommandBuffers	= &command_buffer
	};

	if (vkQueueSubmit(
			application_pointer->transfer_queue, 1, &submit_information, VK_NULL_HANDLE
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_copy_buffer) draw command buffer submition failed" );
		goto out;
	}
	vkQueueWaitIdle( application_pointer->transfer_queue );
	value_to_return = true;

out:
	vkFreeCommandBuffers( application_pointer->device, command_pool, 1, &command_buffer );
	return value_to_return;
}

static bool vsr_create_buffer(
		struct VSR_Application * restrict application_pointer, VkDeviceSize size,
		VkBufferUsageFlags usage,
		const VkMemoryPropertyFlags * restrict properties_lists[static 2],
		int properties_required_amount, VkBuffer * restrict out_buffer_pointer,
		VkDeviceMemory * out_buffer_memory_pointer,
		const char * restrict function_name_pointer
	)
{
	assert_m( function_name_pointer		!= NULL, "No function name found"	);
	assert_m( application_pointer		!= NULL, "No application found"		);
	assert_m( out_buffer_memory_pointer	!= NULL, "No buffer memory found"	);
	assert_m( out_buffer_pointer		!= NULL, "No buffer found"			);

	struct VkBufferCreateInfo buffer_create_information = {
		.sType			= VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size			= size,
		.usage			= usage,
		.sharingMode	= VK_SHARING_MODE_EXCLUSIVE
	};

	if (vkCreateBuffer(
			application_pointer->device, &buffer_create_information, NULL, out_buffer_pointer
		) != VK_SUCCESS )
	{
		woem_push( "(%s) buffer creation failed", function_name_pointer );
		return false;
	}

	VkMemoryRequirements memory_requirements;
	vkGetBufferMemoryRequirements(
		application_pointer->device, *out_buffer_pointer, &memory_requirements
	);

	struct VkMemoryAllocateInfo allocate_information;

	uint32_t memory_type_index = 0;
	if( vsr_find_memory_type(
			application_pointer, memory_requirements.memoryTypeBits, properties_lists[0],
			properties_lists[1], properties_required_amount, &memory_type_index
		) == false )
	{
		woem_push( "(%s) suitable memory type not found", function_name_pointer );
		goto cleanup;
	}

	allocate_information = (struct VkMemoryAllocateInfo) {
		.sType				= VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize		= memory_requirements.size,
		.memoryTypeIndex	= memory_type_index
	};

	if (vkAllocateMemory(
			application_pointer->device, &allocate_information, NULL, out_buffer_memory_pointer
		) != VK_SUCCESS )
	{
		woem_push( "(%s) buffer memory allocation failed", function_name_pointer );
		goto cleanup;
	}

	if( vkBindBufferMemory(
			application_pointer->device, *out_buffer_pointer, *out_buffer_memory_pointer, 0
		) != VK_SUCCESS )
	{
		woem_push( "(%s) buffer memory binding failed", function_name_pointer );
		goto cleanup_memory;
	}

	return true;

cleanup_memory:
	vkFreeMemory( application_pointer->device, *out_buffer_memory_pointer, NULL );
	*out_buffer_memory_pointer = VK_NULL_HANDLE;

cleanup:
	vkDestroyBuffer( application_pointer->device, *out_buffer_pointer, NULL );
	*out_buffer_pointer = VK_NULL_HANDLE;
	return false;
}

static bool vsr_find_memory_type(
		struct VSR_Application * restrict application_pointer, uint32_t type_filter,
		const VkMemoryPropertyFlags * restrict properties_list_required,
		const VkMemoryPropertyFlags * restrict properties_list_forbidden,
		int properties_required_amount, uint32_t * restrict out_type_index_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	VkPhysicalDeviceMemoryProperties * memory_properties_pointer =
		&application_pointer->memory_properties;
	VkDeviceSize candidate_size_best = 0;
	int64_t candidate_priority_best = -1;
	uint32_t candidate_best = UINT32_MAX;

	for ( uint32_t candidate = 0;
			candidate < memory_properties_pointer->memoryTypeCount; ++candidate )
	{
		if( (type_filter & ( (uint32_t) 1 << candidate)) == 0 )
			continue;

		VkMemoryPropertyFlags candidate_flags_current =
			memory_properties_pointer->memoryTypes[candidate].propertyFlags;

		uint32_t candidate_priority_current = UINT32_MAX;

		int64_t required_index_worst = (candidate_priority_best != -1)
			? candidate_priority_best
			: properties_required_amount - 1;
		for ( int64_t required_index_best = 0;
				required_index_best <= required_index_worst; ++required_index_best)
		{
			if( (candidate_flags_current & properties_list_required[required_index_best])
				== properties_list_required[required_index_best])
			{
				if((candidate_flags_current & properties_list_forbidden[required_index_best]
					) ==0)
				{
					candidate_priority_current = (uint32_t) required_index_best;
					break;
				}
			}
		}

		if( candidate_priority_current == UINT32_MAX )
			continue;

		uint32_t heap_index = memory_properties_pointer->memoryTypes[candidate].heapIndex;
		VkDeviceSize candidate_size_current =
			memory_properties_pointer->memoryHeaps[heap_index].size;

		if( candidate_priority_best == -1 ||
			candidate_priority_current < candidate_priority_best ||
			(candidate_priority_current == candidate_priority_best &&
			candidate_size_current > candidate_size_best) )
		{
			candidate_priority_best = candidate_priority_current;
			candidate_size_best = candidate_size_current;
			candidate_best = candidate;
		}
	}

	if( candidate_priority_best == -1 )
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

static struct VkVertexInputAttributeDescription * vsr_get_attribute_descriptions(
		uint32_t * restrict out_size_pointer
	)
{
	assert_m( out_size_pointer != NULL, "No size storage found" );

	size_t size_temporary = 2;

	struct VkVertexInputAttributeDescription * attribute_descriptions_pointer;
	if( sa_malloc_array(
			&attribute_descriptions_pointer, size_temporary,
			sizeof(*attribute_descriptions_pointer)
		) == false )
	{
		ep_exit_print( "(vsr_get_attribute_descriptions) allocation size overflow" );
	}
	else if ( attribute_descriptions_pointer == NULL ) {
		woem_push(
			"(vsr_get_attribute_descriptions) "
			"vertex attribute description memory allocation failed"
		);
	} else {
		memcpy( attribute_descriptions_pointer,
				(struct VkVertexInputAttributeDescription[2]) {
					{
		/*.location	*/	0,
		/*.binding	*/	0,
		/*.format	*/	VK_FORMAT_R32G32_SFLOAT,
		/*.offset	*/	offsetof( struct VSR_Vertex, position )
					},
					{
		/*.location	*/	1,
		/*.binding	*/	0,
		/*.format	*/	VK_FORMAT_R32G32B32_SFLOAT,
		/*.offset	*/	offsetof( struct VSR_Vertex, color )
					}
				},
				size_temporary * sizeof(struct VkVertexInputAttributeDescription)
		);
		if ( out_size_pointer != NULL )
			*out_size_pointer = (uint32_t) size_temporary;
	}

	return attribute_descriptions_pointer;
}

static void vsr_recreate_swap_chain( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	if ( application_pointer->width == 0 || application_pointer->height == 0 )
		return;

	if ( application_pointer->swap_chain_recreate_failed_amount >=
			VSR_LIMIT_SWAPCHAIN_RECREATE_FAILURES )
	{
		VSR_DEBUG_LOG( "Fatal error: swap chain recreation failed" );

		application_pointer->is_running = false;
		glfwSetWindowShouldClose( application_pointer->window_pointer, GLFW_TRUE );
		glfwPostEmptyEvent();

		return;
	}

	VkSwapchainKHR	swap_chain_new;
	VkImage			* swap_chain_images_pointer_new;
	uint32_t		swap_chain_image_views_amount_new;
	VkFormat		swap_chain_image_format_new;
	VkExtent2D		swap_chain_extent_new;
	const char		* error_message_pointer;
	VkImageView		* swap_chain_image_views_pointer_new;
	VkFramebuffer	swap_chain_frame_buffer_new = VK_NULL_HANDLE;

	if( vsr_create_swap_chain(
			application_pointer, &swap_chain_new, &swap_chain_images_pointer_new,
			&swap_chain_image_views_amount_new, &swap_chain_image_format_new,
			&swap_chain_extent_new, &error_message_pointer
		) == false )
		goto cleanup;

	if( vsr_create_image_views(
			application_pointer, swap_chain_images_pointer_new,
			swap_chain_image_views_amount_new, swap_chain_image_format_new,
			&swap_chain_image_views_pointer_new, &error_message_pointer
		) == false )
		goto cleanup_swap_chain;

	if ( vsr_create_frame_buffer(
			application_pointer, swap_chain_image_format_new, swap_chain_extent_new,
			&swap_chain_frame_buffer_new, &error_message_pointer
		) == false )
		goto cleanup_image_views;

	if ( vsr_deletion_cleanup_swap_chain(
			application_pointer, application_pointer->swap_chain,
			application_pointer->swap_chain_image_views_pointer,
			application_pointer->swap_chain_image_views_amount,
			application_pointer->swap_chain_frame_buffer, &error_message_pointer
		) == false )
		goto cleanup_image_views;

	if ( application_pointer->swap_chain_images_pointer != NULL )
		free( application_pointer->swap_chain_images_pointer );

	application_pointer->swap_chain						= swap_chain_new;
	application_pointer->swap_chain_extent				= swap_chain_extent_new;
	application_pointer->swap_chain_image_format		= swap_chain_image_format_new;
	application_pointer->swap_chain_frame_buffer		= swap_chain_frame_buffer_new;
	application_pointer->swap_chain_images_pointer		= swap_chain_images_pointer_new;
	application_pointer->swap_chain_image_views_amount	= swap_chain_image_views_amount_new;
	application_pointer->swap_chain_image_views_pointer	= swap_chain_image_views_pointer_new;

	application_pointer->is_swap_chain_valid = true;
	application_pointer->swap_chain_recreate_failed_amount = 0;

	return;

cleanup_image_views:
	for ( uint32_t image = 0; image < swap_chain_image_views_amount_new; ++image ) {
		if ( swap_chain_image_views_pointer_new[image] != VK_NULL_HANDLE )
			vkDestroyImageView(
				application_pointer->device, swap_chain_image_views_pointer_new[image], NULL
			);
	}
	free( swap_chain_image_views_pointer_new );
	swap_chain_image_views_pointer_new = NULL;

cleanup_swap_chain:
	vkDestroySwapchainKHR( application_pointer->device, swap_chain_new, NULL );
	free( swap_chain_images_pointer_new );

cleanup:
	++application_pointer->swap_chain_recreate_failed_amount;
	VSR_DEBUG_LOGF( "Eror: %s", error_message_pointer );
	return;
}

static bool vsr_create_surface( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	if(glfwCreateWindowSurface(
			application_pointer->instance, application_pointer->window_pointer, NULL,
			&application_pointer->surface
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_create_surface) window surface creation failed" );
		return false;
	}
	return true;
}

static bool vsr_create_logical_device(
		struct VSR_Application * restrict application_pointer,
		const char * restrict * restrict out_error_message_pointer
	)
{
	assert_m( application_pointer		!= NULL, "No application found" );
	assert_m( out_error_message_pointer	!= NULL, "No error message storage found"	);

	/* present, graphic, transfer families at maximum */
	uint32_t unique_families_array[3];
	uint32_t families_amount = 0;

	unique_families_array[families_amount++] =
		application_pointer->queue_family_indices.graphics_family;
	if ( application_pointer->queue_family_indices.graphics_family !=
			application_pointer->queue_family_indices.present_family )
		unique_families_array[families_amount++] =
			application_pointer->queue_family_indices.present_family;

	if ( application_pointer->queue_family_indices.graphics_family !=
			application_pointer->queue_family_indices.transfer_family )
		unique_families_array[families_amount++] =
			application_pointer->queue_family_indices.transfer_family;

	struct VkDeviceQueueCreateInfo * queue_create_informations_array;
	if( sa_malloc_array(
			&queue_create_informations_array, families_amount,
			sizeof(*queue_create_informations_array)
		) == false )
	{
		ep_exit_print( "(vsr_create_logical_device) allocation size overflow" );
	}
	else if ( queue_create_informations_array == NULL ) {
		*out_error_message_pointer =
			"(vsr_create_logical_device) queue create information failed to allocate";
		return false;
	}

	/* priority in [ 0.0f; 1.0f ] */
	const float queue_priority = 1.0f;
	for(size_t queue_family_index = 0;
			queue_family_index < families_amount; ++queue_family_index)
	{
		queue_create_informations_array[queue_family_index] = (struct VkDeviceQueueCreateInfo) {
			.sType				= VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex	= unique_families_array[queue_family_index],
			.queueCount			= 1,
			.pQueuePriorities	= &queue_priority
		};
	}

	struct VkPhysicalDeviceFeatures device_features = { 0 };

	VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swap_chains_maintenance1_features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT,
		.swapchainMaintenance1 = VK_TRUE
	};

	VkPhysicalDeviceImagelessFramebufferFeaturesKHR imageless_features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES_KHR,
		.pNext					= &swap_chains_maintenance1_features,
		.imagelessFramebuffer	= VK_TRUE
	};

	struct VkDeviceCreateInfo create_information = {
		.sType					= VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext					= &imageless_features,
		.queueCreateInfoCount	= (uint32_t) families_amount,
		.pQueueCreateInfos		= queue_create_informations_array,
		.enabledExtensionCount	= device_extensions_amount,
		.ppEnabledExtensionNames= device_extensions_array,
		.pEnabledFeatures		= &device_features
	};
#ifndef NDEBUG
	create_information.enabledLayerCount	= validation_layers_amount;
	create_information.ppEnabledLayerNames	= validation_layers_array;
#endif

	if( vkCreateDevice(
			application_pointer->device_physical, &create_information, NULL,
			&application_pointer->device
		) != VK_SUCCESS )
	{
		*out_error_message_pointer =
			"(vsr_create_logical_device) logical device failed to create";
		free( queue_create_informations_array );
		return false;
	}

	free( queue_create_informations_array );
	vkGetDeviceQueue(
		application_pointer->device, application_pointer->queue_family_indices.graphics_family,
		0, &application_pointer->graphics_queue
	);
	vkGetDeviceQueue(
		application_pointer->device, application_pointer->queue_family_indices.present_family,
		0, &application_pointer->present_queue
	);
	if ( application_pointer->queue_family_indices.has_transfer_family == true ) {
		vkGetDeviceQueue(
			application_pointer->device,
			application_pointer->queue_family_indices.transfer_family, 0,
			&application_pointer->transfer_queue
		);
	} else {
		application_pointer->transfer_queue = application_pointer->graphics_queue;
	}

	return true;
}

static bool vsr_pick_physical_device( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	uint32_t devices_amount = 0;
	vkEnumeratePhysicalDevices( application_pointer->instance, &devices_amount, NULL );

	if ( devices_amount == 0 ) {
		woem_push( "(vsr_pick_physical_device) no GPUs with Vulkan support was found" );
		return false;
	}

	VkPhysicalDevice * devices_pointer;
	if( sa_malloc_array( &devices_pointer, devices_amount, sizeof(*devices_pointer) ) == false )
	{
		ep_exit_print( "(vsr_pick_physical_device) allocation size overflow" );
	}
	else if ( devices_pointer == NULL ) {
		woem_push( "(vsr_pick_physical_device) device memory allocation failed" );
		return false;
	}

	vkEnumeratePhysicalDevices(application_pointer->instance, &devices_amount, devices_pointer);

	struct GPU_choose {
		uint32_t			score;
		VkPhysicalDevice	device;
	};

	struct GPU_choose best_gpu = { 0, VK_NULL_HANDLE };
	for ( uint32_t device_current = 0; device_current < devices_amount; ++device_current ) {
		uint32_t current_score = vsr_rate_device_suitability(
			application_pointer->surface, devices_pointer[device_current]
		);
		if ( current_score > best_gpu.score )
			best_gpu = (struct GPU_choose) {
				current_score,
				devices_pointer[device_current]
			};
	}
	free( devices_pointer );

	if ( best_gpu.score == 0 ) {
		woem_push( "(vsr_pick_physical_device) suitable GPU wasn't found" );
		return false;
	}

	application_pointer->device_physical = best_gpu.device;
	vkGetPhysicalDeviceMemoryProperties(
		application_pointer->device_physical, &application_pointer->memory_properties
	);
	
	return vsr_find_queue_families(
		application_pointer->surface, application_pointer->device_physical,
		&application_pointer->queue_family_indices
	);
}

static bool vsr_find_queue_families(
		VkSurfaceKHR surface, VkPhysicalDevice device,
		struct VSR_Queue_Family_Indices * restrict out_queue_family_indices_pointer
	)
{
	assert_m(
		out_queue_family_indices_pointer != NULL, "No place to store queue family indices found"
	);
	uint32_t queue_families_amount;
	vkGetPhysicalDeviceQueueFamilyProperties( device, &queue_families_amount, NULL );

	if ( queue_families_amount == 0 ) {
		woem_push( "(vsr_find_queue_families) no queue families found" );
		return false;
	}

	struct VkQueueFamilyProperties queue_families_array[VSR_LIMIT_STACK_FAMILIES];
	struct VkQueueFamilyProperties * queue_families_pointer;

	if ( queue_families_amount <= VSR_LIMIT_STACK_FAMILIES ) {
		queue_families_pointer = queue_families_array;
	} else {
		if( sa_malloc_array(
				&queue_families_pointer, queue_families_amount, sizeof(*queue_families_pointer)
			) == false )
		{
			ep_exit_print( "(vsr_find_queue_families) allocation size overflow" );
		}
		else if ( queue_families_pointer == NULL ) {
			woem_push("(vsr_find_queue_families) Queue families data allocation failed");
			return false;
		}
	}
	bool memory_was_dynamically_allocated = (queue_families_pointer != queue_families_array);

	vkGetPhysicalDeviceQueueFamilyProperties(
		device, &queue_families_amount, queue_families_pointer
	);

	struct VSR_Queue_Family_Indices indices = { 0 };
	for ( uint32_t queue_family_index = 0;
			queue_family_index < queue_families_amount; ++queue_family_index )
	{
		if ( queue_families_pointer[queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT ) {
			indices.graphics_family = queue_family_index;
			indices.has_graphics_family = true;
		}

			VkBool32 present_family_supported = false;
			if (vkGetPhysicalDeviceSurfaceSupportKHR(
					device, queue_family_index, surface, &present_family_supported
				) != VK_SUCCESS)
			{
				if ( memory_was_dynamically_allocated == true )
					free( queue_families_pointer );
				woem_push(
					"(vsr_find_queue_families) physical device surface support failed to get"
				);
				return false;
			}
			if ( present_family_supported == true ) {
				indices.present_family = queue_family_index;
				indices.has_present_family = true;
			}

			if ( (queue_families_pointer[queue_family_index].queueFlags & VK_QUEUE_TRANSFER_BIT)
					!= 0 )
			{
				if ( (queue_families_pointer[queue_family_index].queueFlags &
						( VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT )) == 0 )
				{
					indices.transfer_family = queue_family_index;
					indices.has_transfer_family = true;
				} else if ( indices.has_transfer_family == false ) {
					indices.transfer_family = queue_family_index;
					indices.has_transfer_family = true;
				}
			}

			if ( vsr_queue_family_indices_is_complete( &indices ) == true )
				break;
	}
	if ( memory_was_dynamically_allocated == true )
		free( queue_families_pointer );

	*out_queue_family_indices_pointer = indices;
	return true;
}

static bool vsr_queue_family_indices_is_complete(
		struct VSR_Queue_Family_Indices * restrict queue_family_indices_pointer
	)
{
	assert_m( queue_family_indices_pointer != NULL, "No queue family indices found" );

	return
		(queue_family_indices_pointer->has_graphics_family == true &&
		queue_family_indices_pointer->has_present_family == true &&
		queue_family_indices_pointer->has_transfer_family == true);
}

static uint32_t vsr_rate_device_suitability( VkSurfaceKHR surface, VkPhysicalDevice device ) {
	VkPhysicalDeviceProperties device_properties;
	VkPhysicalDeviceFeatures device_features;
	vkGetPhysicalDeviceProperties( device, &device_properties );
	vkGetPhysicalDeviceFeatures( device, &device_features );

	struct VSR_Queue_Family_Indices indices;
	if( vsr_find_queue_families( surface, device, &indices ) == false || 
		vsr_queue_family_indices_is_complete( &indices ) == false )
		return 0;

	if ( vsr_check_device_extensions_support( device ) == false )
		return 0;

	struct VSR_Swap_Chain_Support_Details swap_chain_support;
	if ( vsr_query_swap_chain_support( surface, device, &swap_chain_support ) == false )
		return 0;

	vsr_free_swap_chain_support_details( &swap_chain_support );

	if(	swap_chain_support.formats_amount == 0 && swap_chain_support.present_modes_amount == 0 )
		return 0;

	uint32_t score = 0;
	if ( device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU )
		score += 1000;

	score += device_properties.limits.maxImageDimension2D;
	return score;
}

static void vsr_free_swap_chain_support_details(
		struct VSR_Swap_Chain_Support_Details * restrict swap_chain_support_pointer
	)
{
	if ( swap_chain_support_pointer == NULL )
		return;
	if ( swap_chain_support_pointer->surface_formats_pointer ) {
		free( swap_chain_support_pointer->surface_formats_pointer );
		swap_chain_support_pointer->surface_formats_pointer = NULL;
	}
	if ( swap_chain_support_pointer->present_modes_pointer ) {
		free( swap_chain_support_pointer->present_modes_pointer );
		swap_chain_support_pointer->present_modes_pointer = NULL;
	}
}

static bool vsr_query_swap_chain_support(
		VkSurfaceKHR surface, VkPhysicalDevice device,
		struct VSR_Swap_Chain_Support_Details * restrict out_swap_chain_support_details_pointer
	)
{
	if( vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
			device, surface, &out_swap_chain_support_details_pointer->surface_capabilities
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_query_swap_chain_support) surface capabilities failed to get" );
		return false;
	}

	uint32_t formats_amount;
	if(vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formats_amount, NULL)
		!= VK_SUCCESS)
	{
		woem_push( "(vsr_query_swap_chain_support) surface formats failed to get amount" );
		return false;
	}

	if ( formats_amount != 0 ) {
		if( sa_malloc_array(
				&out_swap_chain_support_details_pointer->surface_formats_pointer,
				formats_amount,
				sizeof(*out_swap_chain_support_details_pointer->surface_formats_pointer)
			) == false )
		{
			ep_exit_print( "(vsr_query_swap_chain_support) allocation size overflow" );
		}
		else if ( out_swap_chain_support_details_pointer->surface_formats_pointer == NULL ) {
			woem_push( "(vsr_query_swap_chain_support) surface format allocation failed" );
			return false;
		}

		if ( vkGetPhysicalDeviceSurfaceFormatsKHR(
				device, surface, &formats_amount,
				out_swap_chain_support_details_pointer->surface_formats_pointer
			) != VK_SUCCESS )
		{
			vsr_free_swap_chain_support_details( out_swap_chain_support_details_pointer );
			woem_push( "(vsr_query_swap_chain_support) surface formats failed to get" );
			return false;
		}
	}
	out_swap_chain_support_details_pointer->formats_amount = formats_amount;

	uint32_t present_modes_amount;
	if ( vkGetPhysicalDeviceSurfacePresentModesKHR(
			device, surface, &present_modes_amount, NULL
		) != VK_SUCCESS )
	{
		vsr_free_swap_chain_support_details( out_swap_chain_support_details_pointer );
		woem_push("(vsr_query_swap_chain_support) surface present modes failed to get amount");
		return false;
	}

	if ( present_modes_amount != 0 ) {
		if( sa_malloc_array(
				&out_swap_chain_support_details_pointer->present_modes_pointer,
				present_modes_amount,
				sizeof(*out_swap_chain_support_details_pointer->present_modes_pointer)
			) == false )
		{
			ep_exit_print( "(vsr_query_swap_chain_support) allocation size overflow" );
		}
		else if ( out_swap_chain_support_details_pointer->present_modes_pointer == NULL ) {
			vsr_free_swap_chain_support_details( out_swap_chain_support_details_pointer );
			woem_push("(vsr_query_swap_chain_support) present modes allocation failed");
			return false;
		}

		if ( vkGetPhysicalDeviceSurfacePresentModesKHR(
				device, surface, &present_modes_amount,
				out_swap_chain_support_details_pointer->present_modes_pointer
			) != VK_SUCCESS)
		{
			vsr_free_swap_chain_support_details( out_swap_chain_support_details_pointer );
			woem_push( "(vsr_query_swap_chain_support) surface present modes failed to get ");
			return false;
		}
	}
	out_swap_chain_support_details_pointer->present_modes_amount = present_modes_amount;

	return true;
}

static bool vsr_synchronization_objects_create(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	if ( sa_calloc_array(
			&application_pointer->synchronization_objects_pointer,
			application_pointer->frames_in_flight_limit,
			sizeof(*application_pointer->synchronization_objects_pointer)
		) == false )
	{
		ep_exit_print( "(vsr_synchronization_objects_create) allocation size overflow" );
	}
	else if ( application_pointer->synchronization_objects_pointer == NULL ) {
		woem_push(
			"(vsr_synchronization_objects_create) synchronization objects allocation failed"
		);
		return false;
	}

	struct VkSemaphoreCreateInfo semaphore_create_information = {
		.sType	= VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};

	struct VkFenceCreateInfo fence_create_information = {
		.sType	= VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags	= VK_FENCE_CREATE_SIGNALED_BIT
	};

	for( uint8_t object_synchronization = 0;
		object_synchronization < application_pointer->frames_in_flight_limit;
		++object_synchronization )
	{
		struct VSR_Synchronization_Objects * synchronization_object_pointer =
			&application_pointer->synchronization_objects_pointer[object_synchronization];
		if (vkCreateSemaphore(
				application_pointer->device, &semaphore_create_information, NULL,
				&synchronization_object_pointer->image_available_semaphore
			) != VK_SUCCESS ||
			vkCreateSemaphore(
				application_pointer->device, &semaphore_create_information, NULL,
				&synchronization_object_pointer->render_finished_semaphore
			) != VK_SUCCESS ||
			vkCreateFence(
				application_pointer->device, &fence_create_information, NULL,
				&synchronization_object_pointer->in_flight_fence
			) != VK_SUCCESS ||
			vkCreateFence(
				application_pointer->device, &fence_create_information, NULL,
				&synchronization_object_pointer->present_fence
			) != VK_SUCCESS)
		{
			woem_push(
				"(vsr_synchronization_objects_create) synchronization objects creation failed"
			);
			vsr_destroy_synchronization_objects(application_pointer, object_synchronization+1);
			free( application_pointer->synchronization_objects_pointer );
			application_pointer->synchronization_objects_pointer = NULL;
			return false;
		}
	}

	return true;
}

static void vsr_destroy_synchronization_objects(
		struct VSR_Application * restrict application_pointer, uint8_t objects_amount
	)
{
	assert_m( application_pointer != NULL, "No application found" );
	assert_m(
		objects_amount <= application_pointer->frames_in_flight_limit,
		"Excess objects to delete"
	);

	for ( uint8_t object_deletion = 0; object_deletion < objects_amount; ++object_deletion ) {
		struct VSR_Synchronization_Objects * synchronization_object_pointer =
			&application_pointer->synchronization_objects_pointer[object_deletion];
		vkDestroySemaphore(
			application_pointer->device,
			synchronization_object_pointer->image_available_semaphore, NULL
		);
		vkDestroySemaphore(
			application_pointer->device,
			synchronization_object_pointer->render_finished_semaphore, NULL
		);
		vkDestroyFence(
			application_pointer->device, synchronization_object_pointer->in_flight_fence, NULL
		);
		vkDestroyFence(
			application_pointer->device, synchronization_object_pointer->present_fence, NULL
		);
	}
}

static void vsr_update_buffer_uniform(
		struct VSR_Application * restrict application_pointer, uint32_t current_image
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	application_pointer->spin_angle_current += application_pointer->spin_angle_rotation;

	if ( application_pointer->is_projection_dirty == true ) {
		float
			width = (application_pointer->width > 0)
				? (float) application_pointer->width
				: 1.f,
			height = (application_pointer->height > 0)
				? (float) application_pointer->height
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
		application_pointer->is_projection_dirty = false;
	}
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
		current_image * application_pointer->buffer_uniform_alignment_size;

	uint8_t * mapped_memory_pointer = (uint8_t *)
		application_pointer->buffers_uniform_mapped_pointer;
	memcpy(
		mapped_memory_pointer + size_offset,
		&buffer_uniform_object,
		application_pointer->buffer_uniform_size
	);

	VkMappedMemoryRange flush_range = {
		.sType	= VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
		.memory	= application_pointer->buffers_uniform_memory,
		.offset	= size_offset,
		.size	= application_pointer->buffer_uniform_alignment_size
	};
	vkFlushMappedMemoryRanges( application_pointer->device, 1, &flush_range );
}

static void vsr_draw_frame( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	/* only works properly because of scaling */
	if ( application_pointer->is_swap_chain_valid == false ) {
		vsr_recreate_swap_chain( application_pointer );
		if ( application_pointer->is_swap_chain_valid == false )
			return;
	}

	const struct VSR_Synchronization_Objects * synchronization_object_pointer =
		&application_pointer->synchronization_objects_pointer
			[application_pointer->current_frame];

	vkWaitForFences(
		application_pointer->device, 1, &synchronization_object_pointer->in_flight_fence,
		VK_TRUE, UINT64_MAX
	);
	vkWaitForFences(
		application_pointer->device, 1, &synchronization_object_pointer->present_fence, VK_TRUE,
		UINT64_MAX
	);

	uint32_t image_index;
	VkResult result;

	while ( true ) {
		result = vkAcquireNextImageKHR(
			application_pointer->device, application_pointer->swap_chain, UINT64_MAX,
			synchronization_object_pointer->image_available_semaphore, VK_NULL_HANDLE,
			&image_index
		);

		if ( result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR ) {
			break;
		} else if ( result == VK_ERROR_OUT_OF_DATE_KHR ) {
			vsr_recreate_swap_chain( application_pointer );
			if ( application_pointer->is_swap_chain_valid == false )
				return;
		} else if ( result == VK_ERROR_SURFACE_LOST_KHR ) {
			vkDestroySurfaceKHR(
				application_pointer->instance, application_pointer->surface, NULL
			);
			application_pointer->surface = NULL;
			if ( vsr_create_surface( application_pointer ) == false )
				ep_exit_print( "(vsr_draw_frame) surface recreation failed" );
			vsr_recreate_swap_chain( application_pointer );
			if ( application_pointer->is_swap_chain_valid == false )
				return;
		} else if ( result == VK_ERROR_DEVICE_LOST ) {
			if ( vsr_device_recreate( application_pointer ) == false ) {
				ep_exit_print( "(vsr_draw_frame) the GPU device has been lost" );
			}
			return;
		} else if ( result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR ) {
			VSR_DEBUG_LOG( "(vsr_draw_frame) acquiring swap chain image failed" );
			application_pointer->is_swap_chain_valid = false;
			return;
		}
	}

	vsr_update_buffer_uniform( application_pointer, application_pointer->current_frame );

	vkResetFences(
		application_pointer->device, 1, &synchronization_object_pointer->in_flight_fence
	);

	/* make sure command buffer is able to be recorded */
	vkResetCommandBuffer(
		application_pointer->command_buffers_pointer[application_pointer->current_frame], 0
	);
	if( vsr_record_command_buffer(
			application_pointer,
			application_pointer->command_buffers_pointer[application_pointer->current_frame],
			image_index
		) == false )
		return;

	VkSemaphore wait_semaphores_array[] = {
		synchronization_object_pointer->image_available_semaphore
	};
	VkPipelineStageFlags wait_stages_array[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

	VkSemaphore signal_semaphores_array[] = {
		synchronization_object_pointer->render_finished_semaphore
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

	if (vkQueueSubmit(
			application_pointer->graphics_queue, 1, &submit_information,
			synchronization_object_pointer->in_flight_fence
		) != VK_SUCCESS )
	{
		VSR_DEBUG_LOG( "(vsr_draw_frame) draw command buffer submition failed" );
		application_pointer->is_swap_chain_valid = false;
		return;
	}

	vkResetFences(
		application_pointer->device, 1, &synchronization_object_pointer->present_fence
	);
	VkSwapchainPresentFenceInfoEXT presentation_fence_information = {
		.sType			= VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
		.swapchainCount	= 1,
		.pFences		= &synchronization_object_pointer->present_fence
	};

	VkSwapchainKHR swap_chains_array[] = { application_pointer->swap_chain };
	struct VkPresentInfoKHR presentation_information = {
		.sType				= VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext				= &presentation_fence_information,
		.waitSemaphoreCount	= 1,
		.pWaitSemaphores	= signal_semaphores_array,
		.swapchainCount		= 1,
		.pSwapchains		= swap_chains_array,
		.pImageIndices		= &image_index
	};

	result = vkQueuePresentKHR( application_pointer->present_queue, &presentation_information );
	if ( result == VK_ERROR_OUT_OF_DATE_KHR ) {
		application_pointer->is_swap_chain_valid = false;
	} else if ( result == VK_ERROR_SURFACE_LOST_KHR ) {
		vkDestroySurfaceKHR(application_pointer->instance, application_pointer->surface, NULL);
		application_pointer->surface = NULL;
		if ( vsr_create_surface( application_pointer ) == false )
			ep_exit_print( "(vsr_draw_frame) surface recreation failed" );
		application_pointer->is_swap_chain_valid = false;
	} else if ( result == VK_ERROR_DEVICE_LOST ) {
		if ( vsr_device_recreate( application_pointer ) == false ) {
			ep_exit_print( "(vsr_draw_frame) the GPU device has been lost" );
		}
		return;
	} else if ( result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR ) {
		VSR_DEBUG_LOGF( "(vsr_draw_frame) swap_chain presentation failed (%d)", result );
		application_pointer->is_swap_chain_valid = false;
	}

	vsr_process_delay_deletion( application_pointer );

	application_pointer->current_frame = (uint8_t)
		((application_pointer->current_frame + 1) %
			application_pointer->frames_in_flight_limit);
}

static bool vsr_record_command_buffer(
		struct VSR_Application * restrict application_pointer, VkCommandBuffer command_buffer,
		uint32_t image_index
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	struct VkCommandBufferBeginInfo command_buffer_begin_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	};

	if( vkBeginCommandBuffer( command_buffer, &command_buffer_begin_information ) != VK_SUCCESS)
	{
		VSR_DEBUG_LOG("(vsr_record_command_buffer) recording command buffer beginning failed");
		return false;
	}

	VkClearValue clear_color = {
		.color = {{ 1.0f, 0.0f, 0.0f, 1.0f }}
	};

	VkRenderPassAttachmentBeginInfoKHR attachment_begin_information = {
		.sType				= VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO_KHR,
		.attachmentCount	= 1,
		.pAttachments		= &application_pointer->swap_chain_image_views_pointer[image_index]
	};

	struct VkRenderPassBeginInfo render_pass_information = {
		.sType				= VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext				= &attachment_begin_information,
		.renderPass			= application_pointer->render_pass,
		.framebuffer		= application_pointer->swap_chain_frame_buffer,
		.renderArea 		= {
			.extent	= application_pointer->swap_chain_extent
		},
		.clearValueCount	= 1, /* values to VK_ATTACHMENT_LOAD_OP_CLEAR */
		.pClearValues		= &clear_color
	};

	vkCmdBeginRenderPass(command_buffer, &render_pass_information, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(
		command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, application_pointer->graphics_pipeline
	);

	struct VkViewport viewport = {
		.width		= (float) application_pointer->swap_chain_extent.width,
		.height		= (float) application_pointer->swap_chain_extent.height,
		.maxDepth	= 1.0f
	};
	vkCmdSetViewport( command_buffer, 0, 1, &viewport );

	struct VkRect2D scissor = {
		.extent	= application_pointer->swap_chain_extent
	};
	vkCmdSetScissor( command_buffer, 0, 1, &scissor );

	VkBuffer vertex_buffers_array[] = { application_pointer->buffer_vertex };
	VkDeviceSize offsets_array[]	= { 0 };
	vkCmdBindVertexBuffers( command_buffer, 0, 1, vertex_buffers_array, offsets_array );
	vkCmdBindIndexBuffer(
		command_buffer, application_pointer->buffer_index, 0, VK_INDEX_TYPE_UINT16
	);

	vkCmdBindDescriptorSets(
		command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		application_pointer->pipeline_layout,
		0,
		1,
		&application_pointer->descriptor_sets_pointer[application_pointer->current_frame],
		0,
		NULL
	);

	vkCmdDrawIndexed(
		command_buffer, (uint32_t)(sizeof(indices_array) / sizeof(indices_array[0])), 1, 0, 0, 0
	);

	vkCmdEndRenderPass( command_buffer );

	if ( vkEndCommandBuffer( command_buffer ) != VK_SUCCESS ) {
		VSR_DEBUG_LOG( "(vsr_record_command_buffer) command buffer recording failed" );
		return false;
	}

	return true;
}

static bool vsr_create_command_buffers( struct VSR_Application * restrict application_pointer )
{
	assert_m( application_pointer != NULL, "No application found" );

	if( sa_malloc_array(
			&application_pointer->command_buffers_pointer,
			application_pointer->frames_in_flight_limit,
			sizeof(*application_pointer->command_buffers_pointer)
		) == false )
	{
		ep_exit_print( "(vsr_create_command_buffers) allocation size overflow" );
	}
	else if ( application_pointer->command_buffers_pointer == NULL ) {
		woem_push( "(vsr_create_command_buffers) command buffer memory allocation failed" );
		return false;
	}

	struct VkCommandBufferAllocateInfo allocation_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool		= application_pointer->command_pool_graphic,
		.level				= VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount	= (uint32_t) application_pointer->frames_in_flight_limit
	};

	if (vkAllocateCommandBuffers(
			application_pointer->device, &allocation_information,
			application_pointer->command_buffers_pointer
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_create_command_buffers) command buffer allocation failed" );
		return false;
	}

	return true;
}

static bool vsr_create_inclusive_command_pool(
		struct VSR_Application * restrict application_pointer,
		VkCommandPool * restrict command_pool_pointer, VkCommandPoolCreateFlags flags,
		uint32_t family
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	struct VkCommandPoolCreateInfo pool_create_information = {
		.sType				= VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags				= flags,
		.queueFamilyIndex	= family
	};

	if (vkCreateCommandPool(
			application_pointer->device, &pool_create_information, NULL, command_pool_pointer
		) != VK_SUCCESS)
	{
		woem_push( "(vsr_create_inclusive_command_pool) command pool creation failed" );
		return false;
	}

	return true;
}

static bool vsr_create_command_pools( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	if( vsr_create_inclusive_command_pool(
			application_pointer,
			&application_pointer->command_pool_graphic,
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			application_pointer->queue_family_indices.graphics_family
		) == false )
		return false;

	if ( application_pointer->queue_family_indices.has_transfer_family == true ) {
		if( vsr_create_inclusive_command_pool(
				application_pointer,
				&application_pointer->command_pool_transfer,
				VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
				application_pointer->queue_family_indices.transfer_family
			) == false )
			return false;
	}

	return true;
}

static bool vsr_create_frame_buffer(
		struct VSR_Application * restrict application_pointer, VkFormat swap_chain_image_format,
		VkExtent2D swap_chain_extent,
		VkFramebuffer * restrict out_swap_chain_frame_buffer_pointer,
		const char * restrict * restrict out_error_message_pointer
	)
{
	assert_m( application_pointer					!= NULL, "No application found"			);
	assert_m( out_swap_chain_frame_buffer_pointer	!= NULL, "No frame buffer storage found");
	assert_m( out_error_message_pointer				!= NULL, "No error message storage found");

	VkFramebufferAttachmentImageInfoKHR attachment_image_information = {
		.sType				= VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO_KHR,
		.usage				= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.width				= swap_chain_extent.width,
		.height				= swap_chain_extent.height,
		.layerCount			= 1,
		.viewFormatCount	= 1,
		.pViewFormats		= &swap_chain_image_format
	};

	VkFramebufferAttachmentsCreateInfoKHR attachment_create_information = {
		.sType						= VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO_KHR,
		.attachmentImageInfoCount	= 1,
		.pAttachmentImageInfos		= &attachment_image_information
	};

	struct VkFramebufferCreateInfo frame_buffer_create_information = {
		.sType				= VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.pNext				= &attachment_create_information,
		.flags				= VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT_KHR,
		.renderPass			= application_pointer->render_pass,
		.attachmentCount	= 1,			/* same with render_pass_create_information */
		.width				= swap_chain_extent.width,
		.height				= swap_chain_extent.height,
		.layers				= 1
	};

	if( vkCreateFramebuffer(
			application_pointer->device, &frame_buffer_create_information, NULL,
			out_swap_chain_frame_buffer_pointer
		) != VK_SUCCESS )
	{
		*out_error_message_pointer = "(vsr_create_frame_buffer) frame_buffer failed to create";
		return false;
	}
	return true;
}

static bool vsr_create_render_pass( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	struct VkAttachmentDescription color_attachment = {
		.format			= application_pointer->swap_chain_image_format,
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
		.colorAttachmentCount	= 1,
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
		.attachmentCount	= 1,
		.pAttachments		= &color_attachment,
		.subpassCount		= 1,
		.pSubpasses			= &subpass,
		.dependencyCount	= 1,
		.pDependencies		= &dependency
	};

	if (vkCreateRenderPass(
			application_pointer->device, &render_pass_create_information, NULL,
			&application_pointer->render_pass
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_create_render_pass) render pass creation failed" );
		return false;
	}

	return true;
}

static bool vsr_create_graphics_pipeline_from_shaders(
		struct VSR_Application * restrict application_pointer,
		const VkShaderModule shader_module_vertex, const VkShaderModule shader_module_fragment
	)
{
	assert_m( application_pointer	!= NULL,			"No application found"				);
	assert_m( shader_module_vertex	!= VK_NULL_HANDLE,	"No vertex shader module found"		);
	assert_m( shader_module_fragment!= VK_NULL_HANDLE,	"No fragment shader module found"	);

	bool value_to_return = false;
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

	uint32_t size = 0;
	struct VkVertexInputAttributeDescription * attribute_descriptions_pointer =
		vsr_get_attribute_descriptions( &size );
	if ( attribute_descriptions_pointer == NULL )
		return false;

	struct VkPipelineVertexInputStateCreateInfo vertex_input_information = {
		.sType						= VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount		= 1,
		.pVertexBindingDescriptions			= &binding_description,
		.vertexAttributeDescriptionCount	= size,
		.pVertexAttributeDescriptions		= attribute_descriptions_pointer
	};

	struct VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType		= VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology	= VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	/* multiple viewports require GPU feature (logical device creation) */
	struct VkViewport viewport = {
		.width		= (float) application_pointer->swap_chain_extent.width,
		.height		= (float) application_pointer->swap_chain_extent.height,
		.maxDepth	= 1.0f
	};

	struct VkRect2D scissor = {
		.offset	= { 0, 0 },
		.extent	= application_pointer->swap_chain_extent
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

	struct VkGraphicsPipelineCreateInfo pipeline_create_information;

	if (vkCreatePipelineLayout(
			application_pointer->device, &pipeline_layout_information, NULL,
			&application_pointer->pipeline_layout
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_create_graphics_pipeline) pipeline layout creation failed" );
		goto cleanup;
	}

	pipeline_create_information = (struct VkGraphicsPipelineCreateInfo) {
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

	if (vkCreateGraphicsPipelines(
			application_pointer->device, VK_NULL_HANDLE, 1, &pipeline_create_information, NULL,
			&application_pointer->graphics_pipeline
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_create_graphics_pipeline) graphics pipeline creation failed" );
		vkDestroyPipelineLayout(
			application_pointer->device, application_pointer->pipeline_layout, NULL
		);
		application_pointer->pipeline_layout = VK_NULL_HANDLE;
		goto cleanup;
	}

	value_to_return = true;

cleanup:
	free( attribute_descriptions_pointer );
	return value_to_return;
}

static bool vsr_create_graphics_pipeline(
		struct VSR_Application * restrict application_pointer
	)
{
	assert_m( application_pointer != NULL, "No application found" );

	bool value_to_return = false;
	VkShaderModule
		shader_module_vertex = VK_NULL_HANDLE,
		shader_module_fragment = VK_NULL_HANDLE;

	size_t file_size = 0;
	char * shader_code_vertex = NULL, * shader_code_fragment = NULL;

	if( hf_file_read( "shaders/vertex.spv", &shader_code_vertex, &file_size ) != 0 ) {
		woem_push( "(vsr_create_graphics_pipeline) vertex shader code failed to get" );
		return false;
	}
	if( vsr_create_shader_module(
			application_pointer->device, shader_code_vertex, file_size, &shader_module_vertex
		) == false)
		goto cleanup;

	if( hf_file_read( "shaders/fragment.spv", &shader_code_fragment, &file_size ) != 0 ) {
		woem_push( "(vsr_create_graphics_pipeline) fragment shader code failed to get" );
		goto cleanup;
	}

	if( vsr_create_shader_module(
			application_pointer->device, shader_code_fragment, file_size,
			&shader_module_fragment
		) == false )
		goto cleanup;

	if( vsr_create_graphics_pipeline_from_shaders(
			application_pointer, shader_module_vertex, shader_module_fragment
		) == false )
		goto cleanup;

	value_to_return = true;

cleanup:
	if( shader_module_vertex != VK_NULL_HANDLE )
		vkDestroyShaderModule( application_pointer->device, shader_module_vertex, NULL );
	if( shader_module_fragment != VK_NULL_HANDLE )
		vkDestroyShaderModule( application_pointer->device, shader_module_fragment, NULL );

	return value_to_return;
}

static bool vsr_create_shader_module(
		VkDevice device, char * restrict shader_code_source_pointer, size_t file_size,
		VkShaderModule * restrict out_shader_module_pointer
	)
{
	assert_m( device					!= VK_NULL_HANDLE,	"No device found"				);
	assert_m( shader_code_source_pointer!= NULL,			"No shadder source found"		);
	assert_m( file_size					> 0,				"No file size found"			);
	assert_m( out_shader_module_pointer	!= NULL,			"No shader module storage found");

	if ( file_size % sizeof(uint32_t) != 0 ) {
		woem_push(
			"(vsr_create_shader_module) "
			"file size is not a multiple of 4, SPIR-V shader file is corrupted" );
		free( shader_code_source_pointer );
		return false;
	}

	uint32_t * aligned_shader_source_code_pointer;
	if ( am_aligned_malloc(
			&aligned_shader_source_code_pointer, sizeof(*aligned_shader_source_code_pointer),
			file_size
		) == false )
	{
		free( shader_code_source_pointer );
		ep_exit_print( "(vsr_create_shader_module) buffer allocation cause overflow" );
	} else if ( aligned_shader_source_code_pointer == NULL ) {
		free( shader_code_source_pointer );
		woem_push( "(vsr_create_shader_module) aligned code failed to allocate" );
		return false;
	}

	memcpy( aligned_shader_source_code_pointer, shader_code_source_pointer, file_size );
	free( shader_code_source_pointer );

	VkShaderModuleCreateInfo create_information = {
		.sType		= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize	= file_size,
		.pCode		= aligned_shader_source_code_pointer
	};

	if (vkCreateShaderModule( device, &create_information, NULL, out_shader_module_pointer )
		!= VK_SUCCESS )
		woem_push( "(vsr_create_shader_module) shader module creation failed" ); 

	am_aligned_free( aligned_shader_source_code_pointer );
	return (*out_shader_module_pointer != VK_NULL_HANDLE);
}

static bool vsr_create_image_views(
		struct VSR_Application * restrict application_pointer,
		VkImage * restrict swap_chain_images_pointer, uint32_t swap_chain_image_views_amount,
		VkFormat swap_chain_image_format,
		VkImageView ** out_swap_chain_image_views_pointer,
		const char * restrict * restrict out_error_message_pointer
	)
{
	assert_m( application_pointer				!= NULL, "No application found"			);
	assert_m( swap_chain_images_pointer			!= NULL, "No swap chain storage found"	);
	assert_m( out_swap_chain_image_views_pointer!= NULL, "No images storage found"		);
	assert_m( out_error_message_pointer			!= NULL,"No error message storage found");

	if ( sa_calloc_array(
			out_swap_chain_image_views_pointer, swap_chain_image_views_amount,
			sizeof(**out_swap_chain_image_views_pointer)
		) == false )
	{
		ep_exit_print( "(vsr_create_image_views) allocation size overflow" );
	}
	else if ( *out_swap_chain_image_views_pointer == NULL ) {
		*out_error_message_pointer =
			"(vsr_create_image_views) image views memory allocation failed";
		return false;
	}

	for ( size_t image = 0; image < swap_chain_image_views_amount; ++image ) {
		VkImageViewCreateInfo create_information = {
			.sType		= VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image		= swap_chain_images_pointer[image],
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
		if (vkCreateImageView(
				application_pointer->device, &create_information, NULL,
				&(*out_swap_chain_image_views_pointer)[image]
			) != VK_SUCCESS )
		{
			for ( uint32_t image_to_delete = 0; image_to_delete < image; ++image_to_delete )
				vkDestroyImageView(
					application_pointer->device,
					(*out_swap_chain_image_views_pointer)[image_to_delete], NULL
				);
			free( *out_swap_chain_image_views_pointer );
			*out_swap_chain_image_views_pointer = NULL;
			*out_error_message_pointer = "(vsr_create_image_views) image views creation failed";
			return false;
		}
	}

	return true;
}

static bool vsr_create_swap_chain(
		struct VSR_Application * restrict application_pointer,
		VkSwapchainKHR * restrict out_swap_chain_pointer,
		VkImage ** out_swap_chain_images_pointer,
		uint32_t * restrict out_swap_chain_image_views_amount_pointer,
		VkFormat * restrict out_swap_chain_image_format_pointer,
		VkExtent2D * restrict out_swap_chain_extent_pointer,
		const char * restrict * restrict out_error_message_pointer
	)
{
	assert_m( application_pointer					!= NULL, "No application found"			);
	assert_m( out_swap_chain_pointer				!= NULL, "No swap chain storage found"	);
	assert_m( out_swap_chain_images_pointer			!= NULL, "No image storage found"		);
	assert_m( out_swap_chain_image_format_pointer	!= NULL, "No format storage found"		);
	assert_m( out_swap_chain_extent_pointer			!= NULL, "No extent storage found"		);
	assert_m( out_error_message_pointer				!= NULL, "No error message storage found");
	assert_m(
		out_swap_chain_image_views_amount_pointer	!= NULL, "No images amount storage found"
	);

	struct VSR_Swap_Chain_Support_Details swap_chain_support;
	if( vsr_query_swap_chain_support(
			application_pointer->surface, application_pointer->device_physical,
			&swap_chain_support
		) == false )
	{
		*out_error_message_pointer = "(vsr_create_swap_chain) no query support";
		return false;
	} else if ( swap_chain_support.surface_capabilities.maxImageExtent.width == 0 ||
		swap_chain_support.surface_capabilities.maxImageExtent.height == 0 )
	{
		*out_error_message_pointer = "(vsr_create_swap_chain) zero surface extent";
		return false;
	}

	struct VkSurfaceFormatKHR surface_format = vsr_choose_swap_surface_format(
			swap_chain_support.surface_formats_pointer, swap_chain_support.formats_amount
		);
	*out_swap_chain_extent_pointer = vsr_choose_swap_extent(
		&swap_chain_support.surface_capabilities, application_pointer->window_pointer
	);

	if( (*out_swap_chain_extent_pointer).width == 0 ||
		(*out_swap_chain_extent_pointer).height== 0 )
	{
		*out_error_message_pointer = "(vsr_create_swap_chain) zero extent";
		return false;
	}

	*out_swap_chain_image_format_pointer = surface_format.format;
	*out_swap_chain_image_views_amount_pointer =
			(swap_chain_support.surface_capabilities.maxImageCount == 0)
		? swap_chain_support.surface_capabilities.minImageCount + 1
		: (uint32_t) cv_clamp_int64_t(
			(uint64_t) swap_chain_support.surface_capabilities.minImageCount + 1,
			(uint64_t) swap_chain_support.surface_capabilities.minImageCount,
			(uint64_t) swap_chain_support.surface_capabilities.maxImageCount);

	VkSwapchainPresentScalingCreateInfoEXT scaling_create_information = {
		.sType				= VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT,
		.scalingBehavior	= VK_PRESENT_SCALING_STRETCH_BIT_EXT,
		.presentGravityX	= VK_PRESENT_GRAVITY_MIN_BIT_EXT,
		.presentGravityY	= VK_PRESENT_GRAVITY_MIN_BIT_EXT
	};

	VkSwapchainKHR old_swap_chain = application_pointer->swap_chain;
	struct VkSwapchainCreateInfoKHR create_information = {
		.sType				= VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.pNext				= &scaling_create_information,
		.surface			= application_pointer->surface,
		.minImageCount		= *out_swap_chain_image_views_amount_pointer,
		.imageFormat		= surface_format.format,
		.imageColorSpace	= surface_format.colorSpace,
		.imageExtent		= *out_swap_chain_extent_pointer,
		.imageArrayLayers	= 1, /* image consist of this amount of layers */
		.imageUsage			= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.preTransform		= swap_chain_support.surface_capabilities.currentTransform,
		.compositeAlpha		= VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode		= VK_PRESENT_MODE_FIFO_KHR,
		.clipped			= VK_TRUE,
		.oldSwapchain		= old_swap_chain
	};
	vsr_free_swap_chain_support_details( &swap_chain_support );

	uint32_t unique_families_array[] = {
		application_pointer->queue_family_indices.graphics_family,
		application_pointer->queue_family_indices.present_family
	};
	if (application_pointer->queue_family_indices.graphics_family !=
		application_pointer->queue_family_indices.present_family )
	{
		create_information.imageSharingMode			= VK_SHARING_MODE_CONCURRENT;
		create_information.queueFamilyIndexCount	= 2;
		create_information.pQueueFamilyIndices		= unique_families_array;
	} else {
		create_information.imageSharingMode			= VK_SHARING_MODE_EXCLUSIVE;
	}

	if (vkCreateSwapchainKHR(
			application_pointer->device, &create_information, NULL, out_swap_chain_pointer
		) != VK_SUCCESS )
	{
		*out_error_message_pointer = "(vsr_create_swap_chain) swap chain failed to create";
		return false;
	}

	if (vkGetSwapchainImagesKHR(
			application_pointer->device, *out_swap_chain_pointer,
			out_swap_chain_image_views_amount_pointer, NULL
		) != VK_SUCCESS )
	{
		*out_error_message_pointer =
			"(vsr_create_swap_chain) image count for swap chain failed to get";
		goto cleanup;
	} else if ( *out_swap_chain_image_views_amount_pointer == 0 ) {
		*out_error_message_pointer = "(vsr_create_swap_chain) no images returned";
		goto cleanup;
	}

	if( sa_malloc_array(
			out_swap_chain_images_pointer,
			*out_swap_chain_image_views_amount_pointer,
			sizeof(**out_swap_chain_images_pointer)
		) == false )
	{
		ep_exit_print( "(vsr_create_swap_chain) allocation size overflow" );
	}
	else if ( *out_swap_chain_images_pointer == NULL ) {
		*out_error_message_pointer =
			"(vsr_create_swap_chain) swap chain images memory allocation failed";
		goto cleanup;
	}

	if (vkGetSwapchainImagesKHR(
			application_pointer->device, *out_swap_chain_pointer,
			out_swap_chain_image_views_amount_pointer,
			*out_swap_chain_images_pointer
		) != VK_SUCCESS )
	{
		*out_error_message_pointer =
			"(vsr_create_swap_chain) swap chain images failed to create";
		goto cleanup_images;
	}

	glm_lookat(
		(vec3){ 2.f, 2.f, 2.f },
		(vec3){ 0.f, 0.f, 0.f },
		(vec3){ 0.f, 0.f, 1.f },
		application_pointer->cached_view
	);

	application_pointer->is_projection_dirty = true;
	return true;

cleanup_images:
	free( *out_swap_chain_images_pointer );
	*out_swap_chain_images_pointer = NULL;

cleanup:
	vkDestroySwapchainKHR( application_pointer->device, *out_swap_chain_pointer, NULL );
	*out_swap_chain_pointer = VK_NULL_HANDLE;
	return false;
}

static struct VkSurfaceFormatKHR vsr_choose_swap_surface_format(
		const struct VkSurfaceFormatKHR * restrict available_formats_pointer,
		size_t available_formats_amount
	)
{
	for ( size_t format_index = 0; format_index < available_formats_amount; ++format_index ) {
		if (available_formats_pointer[format_index].format == VK_FORMAT_B8G8R8A8_SRGB &&
			available_formats_pointer[format_index].
				colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			return available_formats_pointer[format_index];
	}
	return available_formats_pointer[0];
}

static struct VkExtent2D vsr_choose_swap_extent(
		const struct VkSurfaceCapabilitiesKHR * restrict surface_capabilities_pointer,
		GLFWwindow * restrict window_pointer
	)
{
	if ( surface_capabilities_pointer->currentExtent.width != UINT32_MAX )
		return surface_capabilities_pointer->currentExtent;

	int width, height;
	glfwGetFramebufferSize( window_pointer, &width, &height );

	struct VkExtent2D extent = {
		(uint32_t) width,
		(uint32_t) height
	};

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

static bool vsr_check_device_extensions_support( VkPhysicalDevice device ) {
	uint32_t extension_properties_amount;
	if ( vkEnumerateDeviceExtensionProperties(
			device, NULL, &extension_properties_amount, NULL
		) != VK_SUCCESS )
	{
		woem_push(
			"(vsr_check_device_extensions_support) "
			"enumeration the number of device extension properties failed"
		);
		return false;
	}

	if ( device_extensions_amount == 0 )
		return true;

	if ( extension_properties_amount == 0 )
		return false;

	struct VkExtensionProperties * available_extensions_pointer;
	if( sa_malloc_array(
			&available_extensions_pointer, extension_properties_amount,
			sizeof(*available_extensions_pointer)
		) == false )
	{
		ep_exit_print( "(vsr_check_device_extensions_support) allocation size overflow" );
	}
	else if ( available_extensions_pointer == NULL ) {
		woem_push(
			"(vsr_check_device_extensions_support) allocation of extension properties failed"
		);
		return false;
	}

	if ( vkEnumerateDeviceExtensionProperties(
			device, NULL, &extension_properties_amount, available_extensions_pointer
		) != VK_SUCCESS )
	{
		free( available_extensions_pointer );
		woem_push(
			"(vsr_check_device_extensions_support) "
			"Vulkan Enumeration the number of Device Extension Properties failed"
		);
		return false;
	}

	for ( uint32_t device_extension_index = 0;
			device_extensions_amount > device_extension_index; ++device_extension_index )
	{
		bool device_extension_found = false;
		for ( uint32_t device_extension_property_index = 0;
				extension_properties_amount > device_extension_property_index;
				++device_extension_property_index )
		{
			if ( strcmp(
					device_extensions_array[device_extension_index],
					available_extensions_pointer[device_extension_property_index].extensionName
				) == 0)
			{
				device_extension_found = true;
				break;
			}
		}
		if ( device_extension_found == false ) {
			free( available_extensions_pointer );
			return false;
		}
	}
	free( available_extensions_pointer );

	return true;
}

static bool vsr_create_instance( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

#ifndef NDEBUG

	if ( vsr_check_validation_layer_support() == false )
		fprintf( stderr, "(vsr_create_instance) requested validation layers aren't available" );

#endif

	struct VkApplicationInfo application_information = {
		.sType				= VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName	= "Hello Vulkan",
		.applicationVersion = VK_MAKE_VERSION( 1, 0, 0 ),
		.pEngineName		= "No Engine",
		.engineVersion		= VK_MAKE_VERSION( 1, 0, 0 ),
		.apiVersion			= VK_API_VERSION_1_0
	};

	uint32_t extension_amount_required = 0;
	const char ** extensions_required_pointer = vsr_get_required_extensions(
		&extension_amount_required
	);
	if ( extensions_required_pointer == NULL )
		return false;

	uint32_t extension_amount_available = 0;
	struct VkExtensionProperties * extensions_available_pointer = vsr_get_available_extensions(
		&extension_amount_available
	);
	if ( extensions_available_pointer == NULL ) {
		free( extensions_required_pointer );
		return false;
	}

#ifndef NDEBUG
	fprintf(stderr, "required extensions:\n" );
	for ( uint32_t extension_required_index = 0;
			extension_required_index < extension_amount_required; ++extension_required_index )
		fprintf(stderr, "\t%s\n", extensions_required_pointer[extension_required_index] );

	fprintf(stderr, "\navailable extensions:\n" );
	for ( uint32_t extension_available_index = 0;
			extension_available_index < extension_amount_available; ++extension_available_index)
		fprintf( stderr,
			"\t%s\n", extensions_available_pointer[extension_available_index].extensionName
		);
#endif

	for ( uint32_t extension_required_index = 0;
			extension_required_index < extension_amount_required; ++extension_required_index )
	{
		int result = 0;
		for ( uint32_t extension_available_index = 0;
				extension_available_index < extension_amount_available;
				++extension_available_index )
		{
			result = strcmp(
				extensions_required_pointer[extension_required_index],
				extensions_available_pointer[extension_available_index].extensionName
			);
			if ( result == 0 )
				break;
		}
		if ( result != 0 )
			fprintf( stderr,
				"Lack of extension: %s\n", extensions_required_pointer[extension_required_index]
			);
	}
	free( extensions_available_pointer );

	struct VkInstanceCreateInfo create_information = {
		.sType						= VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo			= &application_information,
		.enabledExtensionCount		= extension_amount_required,
		.ppEnabledExtensionNames	= extensions_required_pointer
	};


#ifndef NDEBUG
	struct VkDebugUtilsMessengerCreateInfoEXT debug_create_information;
	vsr_populate_debug_messenger_create_information( &debug_create_information );

	create_information.enabledLayerCount	= (uint32_t) validation_layers_amount;
	create_information.ppEnabledLayerNames	= validation_layers_array;
	create_information.pNext =
		(struct VkDebugUtilsMessengerCreateInfoEXT *) &debug_create_information;
#endif

	if ( vkCreateInstance(
			&create_information, NULL, &application_pointer->instance
		) != VK_SUCCESS )
	{
		free( extensions_required_pointer );
		woem_push( "(vsr_create_instance) failed to create instance" );
		return false;
	}
	free( extensions_required_pointer );
	return true;
}

const char ** vsr_get_required_extensions( uint32_t * restrict glfw_extension_amount_pointer ) {
	assert_m( glfw_extension_amount_pointer, "No extension amount storage found" );
	*glfw_extension_amount_pointer = 0;
	const char ** glfw_extensions_pointer;

	glfw_extensions_pointer = glfwGetRequiredInstanceExtensions(glfw_extension_amount_pointer);

	uint32_t total_size = *glfw_extension_amount_pointer + instance_extensions_amount;

#ifndef NDEBUG
	++total_size;
#endif

	const char ** extensions_pointer;
	if( sa_malloc_array(
			&extensions_pointer, total_size, sizeof( *glfw_extensions_pointer )
		) == false )
	{
		ep_exit_print( "(vsr_get_required_extensions) allocation size overflow" );
	}
	else if ( extensions_pointer == NULL ) {
		woem_push( "(vsr_get_required_extensions) allocation failed" );
		return NULL;
	}

	for ( uint32_t extension_index = 0;
			extension_index < *glfw_extension_amount_pointer; ++extension_index )
		extensions_pointer[extension_index] = glfw_extensions_pointer[extension_index];
	for ( uint32_t extension_index = 0;
			extension_index < instance_extensions_amount; ++extension_index )
	{
		extensions_pointer[extension_index + *glfw_extension_amount_pointer] =
			instance_extensions_array[extension_index];
	}

	*glfw_extension_amount_pointer += instance_extensions_amount;

#ifndef NDEBUG
	extensions_pointer[*glfw_extension_amount_pointer] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
	++*glfw_extension_amount_pointer;
#endif

	return extensions_pointer;
}

static struct VkExtensionProperties * vsr_get_available_extensions(
		uint32_t * restrict extension_amount
	)
{
	assert_m( extension_amount != NULL, "No extension amount storage found" );

	*extension_amount = 0;
	if ( vkEnumerateInstanceExtensionProperties( NULL, extension_amount, NULL ) != VK_SUCCESS )
	{
		woem_push(
			"(vsr_get_available_extensions) "
			"Vulkan Enumaration the Number of Instance Extension Properties failed"
		);
		return NULL;
	}

	struct VkExtensionProperties * extensions;
	if( sa_malloc_array( &extensions, *extension_amount, sizeof(*extensions) ) == false ) {
		ep_exit_print( "(vsr_get_available_extensions) allocation size overflow" );
	}
	else if ( extensions == NULL ) {
		woem_push(
			"(vsr_get_available_extensions) Vulkan Extension Properties allocation failed"
		);
		return NULL;
	}

	if(vkEnumerateInstanceExtensionProperties(NULL, extension_amount, extensions) != VK_SUCCESS)
	{
		free( extensions );
		woem_push(
			"(vsr_get_available_extensions) "
			"Vulkan Enumaration of Instance Extension Properties failed"
		);
		return NULL;
	}
	return extensions;
}

#ifndef NDEBUG
static bool vsr_check_validation_layer_support(void) {
	if ( validation_layers_amount == 0 )
		return true;

	uint32_t instance_layers_amount;
	VkResult result = vkEnumerateInstanceLayerProperties( &instance_layers_amount, NULL );
	if ( result != VK_SUCCESS )
		return false;

	struct VkLayerProperties * available_layers_pointer;
	if( sa_malloc_array(
			&available_layers_pointer, instance_layers_amount, sizeof(*available_layers_pointer)
		) == false )
	{
		ep_exit_print( "(vsr_check_validation_layer_support) allocation size overflow" );
	}
	else if ( available_layers_pointer == NULL ) {
		woem_push( "(vsr_check_validation_layer_support) allocation failed" );
		return false;
	}

	result = vkEnumerateInstanceLayerProperties(
		&instance_layers_amount, available_layers_pointer
	);
	if ( result != VK_SUCCESS ) {
		free( available_layers_pointer );
		return false;
	}
	for ( uint32_t validation_layer_index = 0;
			validation_layer_index < validation_layers_amount; ++validation_layer_index ) {
		bool validation_layer_found = false;
		for ( uint32_t instance_layer_index = 0;
				instance_layers_amount > instance_layer_index; ++instance_layer_index )
		{
			if ( strcmp(
					validation_layers_array[validation_layer_index],
					available_layers_pointer[instance_layer_index].layerName
				) == 0 )
			{
				validation_layer_found = true;
				break;
			}
		}
		if ( validation_layer_found == false ) {
			free( available_layers_pointer );
			return false;
		}
	}
	free( available_layers_pointer );
	return true;
}

static VkResult vsr_create_debug_utils_messenger_extension(
		VkInstance instance,
		const VkDebugUtilsMessengerCreateInfoEXT * restrict create_information_pointer,
		const VkAllocationCallbacks	* restrict allocator_pointer,
		VkDebugUtilsMessengerEXT * restrict debug_messenger_pointer
	) 
{
	PFN_vkCreateDebugUtilsMessengerEXT function = (PFN_vkCreateDebugUtilsMessengerEXT) 
		vkGetInstanceProcAddr( instance, "vkCreateDebugUtilsMessengerEXT" );
	if ( function != NULL )
		return function(
			instance, create_information_pointer, allocator_pointer, debug_messenger_pointer
		);
	else
		return VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void vsr_destroy_debug_utils_messenger_extension(
		VkInstance instance, VkDebugUtilsMessengerEXT debug_messenger,
		const VkAllocationCallbacks * restrict allocator_pointer
	)
{
	PFN_vkDestroyDebugUtilsMessengerEXT function_pointer = (PFN_vkDestroyDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr( instance, "vkDestroyDebugUtilsMessengerEXT" );
	if ( function_pointer != NULL )
		function_pointer( instance, debug_messenger, allocator_pointer );
}

static bool vsr_setup_debug_messenger( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	VkDebugUtilsMessengerCreateInfoEXT create_information;
	vsr_populate_debug_messenger_create_information( &create_information );
	if( vsr_create_debug_utils_messenger_extension(
			application_pointer->instance, &create_information, NULL,
			&application_pointer->debug_messenger_function
		) != VK_SUCCESS )
	{
		woem_push( "(vsr_setup_debug_messenger) failed to set up debug messenger" );
		return false;
	}

	return true;
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

static void vsr_populate_debug_messenger_create_information(
		struct VkDebugUtilsMessengerCreateInfoEXT * restrict creation_information_pointer
	)
{
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
#endif

static bool vsr_window_initialization( struct VSR_Application * restrict application_pointer ) {
	assert_m( application_pointer != NULL, "No application found" );

	if ( glfwInit() != GLFW_TRUE ) {
		woem_push( "(vsr_window_initialization) GLFW initializatino failed" );
		return false;
	}

	glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );

	application_pointer->window_pointer = glfwCreateWindow(
		VSR_WINDOW_WWIDTH, VSR_WINDOW_HEIGHT, "VULKAN SQUARE ROTATION", NULL, NULL
	);
	if ( application_pointer->window_pointer == NULL ) {
		woem_push( "(vsr_window_initialization) window creation failed" );
		glfwTerminate();
		return false;
	}

	glfwSetWindowUserPointer(
		application_pointer->window_pointer, application_pointer
	);
	glfwSetKeyCallback(
		application_pointer->window_pointer, vsr_key_callback
	);
	glfwSetWindowIconifyCallback(
		application_pointer->window_pointer, vsr_window_iconify_callback
	);
	glfwSetFramebufferSizeCallback(
		application_pointer->window_pointer, vsr_frame_buffer_size_callback
	);

	application_pointer->is_initialized_glfw = true;

	return true;
}

static void vsr_window_iconify_callback( GLFWwindow * window_pointer, int iconified ) {
	struct VSR_Application * application_pointer = glfwGetWindowUserPointer( window_pointer );

	pthread_mutex_lock( &application_pointer->render_mutex );

	application_pointer->is_minimized = (iconified == GLFW_TRUE);

	pthread_cond_signal( &application_pointer->render_condition );
	pthread_mutex_unlock( &application_pointer->render_mutex );
}

static void vsr_key_callback(
		GLFWwindow * window_pointer, int key, int scancode, int action, int mods
	)
{
	(void) mods; (void) scancode;
	if ( action == GLFW_PRESS && key == GLFW_KEY_ESCAPE )
		glfwSetWindowShouldClose( window_pointer, GLFW_TRUE );
}

static void vsr_frame_buffer_size_callback( GLFWwindow * window_pointer, int width, int height )
{
	struct VSR_Application * application_pointer = glfwGetWindowUserPointer( window_pointer );

	pthread_mutex_lock( &application_pointer->render_mutex );

	application_pointer->width	= width;
	application_pointer->height	= height;
	if ( width != 0 && height != 0 ) {
		application_pointer->is_projection_dirty = true;
		application_pointer->is_minimized = false;
	}
	else application_pointer->is_minimized = true;

	pthread_cond_signal( &application_pointer->render_condition );
	pthread_mutex_unlock( &application_pointer->render_mutex );
}
