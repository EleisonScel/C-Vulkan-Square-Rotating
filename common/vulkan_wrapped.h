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

#pragma once

#ifndef VULKAN_WRAPPED_H
#define VULKAN_WRAPPED_H

#include "assert_m.h"		/* assert_m */

#include <vulkan/vulkan.h>	/* VkResult */

static inline VkResult vkCreateInstance_wrapped(
		const struct VkInstanceCreateInfo * restrict create_information_pointer,
		const VkAllocationCallbacks * restrict allocator_pointer,
		VkInstance * restrict out_instance_pointer
	)
{
	assert_m( out_instance_pointer		!= NULL, "No instance storage found"	);
	assert_m( create_information_pointer!= NULL, "No create information found"	);

	VkInstance temporary_instance;
	VkResult result = vkCreateInstance(
		create_information_pointer, allocator_pointer, &temporary_instance
	);

	if( result == VK_SUCCESS )
		*out_instance_pointer = temporary_instance;

	return result;
}

static inline VkResult vkCreateDevice_wrapped(
		VkPhysicalDevice device_physical,
		const struct VkDeviceCreateInfo * restrict create_information_pointer,
		const VkAllocationCallbacks * restrict allocator_pointer,
		VkDevice * restrict out_device_pointer
	)
{
	assert_m( device_physical != VK_NULL_HANDLE, "No physical device found"		);
	assert_m( out_device_pointer		!= NULL, "No device storage found"		);
	assert_m( create_information_pointer!= NULL, "No create information found"	);

	VkDevice temporary_device;
	VkResult result = vkCreateDevice(
		device_physical, create_information_pointer, allocator_pointer, &temporary_device
	);

	if( result == VK_SUCCESS )
		*out_device_pointer = temporary_device;

	return result;
}

static inline VkResult vkCreateDescriptorPool_wrapped(
		VkDevice device,
		const struct VkDescriptorPoolCreateInfo * restrict create_information_pointer,
		const VkAllocationCallbacks * restrict allocator_pointer,
		VkDescriptorPool * restrict out_descriptor_pool_pointer
	)
{
	assert_m( device != VK_NULL_HANDLE, "No device found" );
	assert_m( out_descriptor_pool_pointer	!= NULL, "No descriptor pool storage found"	);
	assert_m( create_information_pointer	!= NULL, "No create information found"		);

	VkDescriptorPool temporary_descriptor_pool;
	VkResult result = vkCreateDescriptorPool(
		device, create_information_pointer, allocator_pointer, &temporary_descriptor_pool
	);

	if( result == VK_SUCCESS )
		*out_descriptor_pool_pointer = temporary_descriptor_pool;

	return result;
}

static inline VkResult vkCreateDescriptorSetLayout_wrapped(
		VkDevice device,
		const struct VkDescriptorSetLayoutCreateInfo * restrict create_information_pointer,
		const VkAllocationCallbacks * restrict allocator_pointer,
		VkDescriptorSetLayout * restrict out_descriptor_set_layout_pointer
	)
{
	assert_m( device != VK_NULL_HANDLE, "No device found" );
	assert_m( out_descriptor_set_layout_pointer	!= NULL,"No descriptor set layout storage found");
	assert_m( create_information_pointer		!= NULL,"No create information found"			);

	VkDescriptorSetLayout temporary_descriptor_set_layout;
	VkResult result = vkCreateDescriptorSetLayout(
		device, create_information_pointer, allocator_pointer, &temporary_descriptor_set_layout
	);

	if( result == VK_SUCCESS )
		*out_descriptor_set_layout_pointer = temporary_descriptor_set_layout;

	return result;
}

static inline VkResult vkCreateRenderPass_wrapped(
		VkDevice device,
		const struct VkRenderPassCreateInfo * restrict create_information_pointer,
		const VkAllocationCallbacks * restrict allocator_pointer,
		VkRenderPass * restrict out_render_pass_pointer
	)
{
	assert_m( device != VK_NULL_HANDLE, "No device found" );
	assert_m( out_render_pass_pointer	!= NULL, "No render pass storage found");
	assert_m( create_information_pointer!= NULL, "No create information found" );

	VkRenderPass temporary_render_pass;
	VkResult result = vkCreateRenderPass(
		device, create_information_pointer, allocator_pointer, &temporary_render_pass
	);

	if( result == VK_SUCCESS )
		*out_render_pass_pointer = temporary_render_pass;

	return result;
}

static inline VkResult vkCreatePipelineLayout_wrapped(
		VkDevice device,
		const struct VkPipelineLayoutCreateInfo * restrict create_information_pointer,
		const VkAllocationCallbacks * restrict allocator_pointer,
		VkPipelineLayout * restrict out_pipeline_layout_pointer
	)
{
	assert_m( device != VK_NULL_HANDLE, "No device found" );
	assert_m( out_pipeline_layout_pointer	!= NULL, "No pipeline layout storage found" );
	assert_m( create_information_pointer	!= NULL, "No create information found"		);

	VkPipelineLayout temporary_pipeline_layout;
	VkResult result = vkCreatePipelineLayout(
		device, create_information_pointer, allocator_pointer, &temporary_pipeline_layout
	);

	if( result == VK_SUCCESS )
		*out_pipeline_layout_pointer = temporary_pipeline_layout;

	return result;
}

static inline VkResult vkAllocateMemory_wrapped(
		VkDevice device,
		const VkMemoryAllocateInfo * restrict create_information_pointer,
		const VkAllocationCallbacks * restrict allocator_pointer,
		VkDeviceMemory * restrict out_memory_pointer
	)
{
	assert_m( device != VK_NULL_HANDLE, "No device found" );
	assert_m( out_memory_pointer		!= NULL, "No memory storage found"		);
	assert_m( create_information_pointer!= NULL, "No create information found"	);

	VkDeviceMemory temporary_memory;
	VkResult result = vkAllocateMemory(
		device, create_information_pointer, allocator_pointer, &temporary_memory
	);

	if( result == VK_SUCCESS )
		*out_memory_pointer = temporary_memory;

	return result;
}

static inline VkResult vkCreateCommandPool_wrapped(
		VkDevice device,
		const struct VkCommandPoolCreateInfo * restrict create_information_pointer,
		const VkAllocationCallbacks * restrict allocator_pointer,
		VkCommandPool * restrict out_command_pool_pointer
	)
{
	assert_m( device != VK_NULL_HANDLE, "No device found" );
	assert_m( out_command_pool_pointer	!= NULL, "No command pool storage found");
	assert_m( create_information_pointer!= NULL, "No create information found"	);

	VkCommandPool temporary_command_pool;
	VkResult result = vkCreateCommandPool(
		device, create_information_pointer, allocator_pointer, &temporary_command_pool
	);

	if( result == VK_SUCCESS )
		*out_command_pool_pointer = temporary_command_pool;

	return result;
}

static inline VkResult vkCreateSwapchainKHR_wrapped(
		VkDevice device,
		const struct VkSwapchainCreateInfoKHR * restrict create_information_pointer,
		const VkAllocationCallbacks * restrict allocator_pointer,
		VkSwapchainKHR * restrict out_swap_chain_pointer
	)
{
	assert_m( device != VK_NULL_HANDLE, "No device found" );
	assert_m( out_swap_chain_pointer	!= NULL, "No swap chain storage found" );
	assert_m( create_information_pointer!= NULL, "No create information found" );

	VkSwapchainKHR temporary_swap_chain;
	VkResult result = vkCreateSwapchainKHR(
		device, create_information_pointer, allocator_pointer, &temporary_swap_chain
	);

	if( result == VK_SUCCESS )
		*out_swap_chain_pointer = temporary_swap_chain;

	return result;
}

static inline VkResult vkCreateFramebuffer_wrapped(
		VkDevice device,
		const struct VkFramebufferCreateInfo * restrict create_information_pointer,
		const VkAllocationCallbacks * restrict allocator_pointer,
		VkFramebuffer * restrict out_frame_buffer_pointer
	)
{
	assert_m( device != VK_NULL_HANDLE, "No device found" );
	assert_m( out_frame_buffer_pointer	!= NULL, "No frame buffer storage found");
	assert_m( create_information_pointer!= NULL, "No create information found"	);

	VkFramebuffer temporary_frame_buffer;
	VkResult result = vkCreateFramebuffer(
		device, create_information_pointer, allocator_pointer, &temporary_frame_buffer
	);

	if( result == VK_SUCCESS )
		*out_frame_buffer_pointer = temporary_frame_buffer;

	return result;
}

#endif /* VULKAN_WRAPPED_H */
