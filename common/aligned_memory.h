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
/* File:
 * aligned_memory.h
 *
 * Description:
 * Portable aligned memory allocation, reallocation and free
 */

#ifndef ALIGNED_MEMORY_H
#define ALIGNED_MEMORY_H

#	include "safe_alloc.h"	/* enum sa_allocation_status */

#	include <stddef.h>		/* size_t	*/
#	include <stdbool.h>		/* bool		*/

/* Logic to choose the alignment backend
 *
 * Available backends and its requirements (in order of preferences unless forces):
 * - AM_BACKEND_WINDOWS	: alignment must be a power of two,
 *							. _WIN32,
 *							. C99
 * - AM_BACKEND_POSIX	: alignment must be a power of two and a multiple of a pointer size,
 *							. _POSIX_C_SOURCE >= 200112L
 *							. C99
 * - AM_BACKEND_STANDARD: alignment must be a power of two,
 *							. C11,
 *							. non-Windows system
 * - AM_BACKEND_FALLBACK:
 *		- default (AM_ALIGN_BITWISE = 1)					:
 *								: pointer size and alignment must be a power of two
 *		- with AM_ALLOW_NON_POWER_OF_TWO_ALIGNMENT defined	:
 *								: any non-zero alignment allowed
 *									(uses a slower division-based method)
 *							. uintptr_t
 *							. C99
 *
 * Available user macro definition before including the header:
 * - AM_FORCE_WINDOWS					: force Windows backend
 * - AM_FORCE_POSIX						: force POSIX backend
 * - AM_FORCE_C11						: force C11 backend
 * - AM_FORCE_FALLBACK					: force fallback backend
 * - AM_ALLOW_NON_POWER_OF_TWO_ALIGNMENT: allow non-power-of-two alignment in
 *												(AM_BACKEND_FALLBACK only)
 * - AM_NO_REALLOC						: exclude realloc function and reduce header size
 * - AM_NO_CALLOC						: exclude calloc
 */

/* Function:
 * release a raw memory block allocated with an am_aligned_malloc
 *
 * Parameters:
 * buffer_pointer - pointer to memory block previously allocated with an am_aligned_malloc
 */
void am_aligned_free( void * restrict buffer_pointer );

/* Function:
 * allocate aligned memory block
 *
 * Parameters:
 * out_buffer_pointer	- holder to allocated memory
 * alignment			- desired address alignment
 * size					- size of memory block to allocate in bytes
 *
 * Returns:
 * true					- math is safe, aligned memory allocation attempted
 *							(must be released with am_aligned_free)
 * false				- invalid arguments, overflow
 */
bool am_aligned_malloc( void * restrict out_buffer_pointer, size_t alignment, size_t size );
/* Function:
 * safe version of am_aligned_malloc with array bounds checking
 *
 * Parameters:
 * out_buffer_pointer	- holder to allocated memory
 * alignment			- desired address alignment
 * elements_amount		- new amount of elements
 * element_size			- size of each element
 *
 * Returns:
 * true					- math is safe, aligned memory allocation attempted
 *							(must be released with am_aligned_free)
 * false				- invalid arguments, overflow
 */
bool am_aligned_malloc_array( void * restrict out_buffer_pointer, size_t alignment, size_t elements_amount, size_t element_size );
#	ifndef AM_NO_REALLOC
/* Function:
 * reallocate aligned memory block
 *
 * Precondition:
 * pointer_original			- must be previously allocated with an am_aligned_malloc
 * size_new					- must be positive
 *
 * Parameters:
 * out_buffer_pointer		- holder to allocated memory
 * pointer_original			- existing aligned memory block
 * size_new					- size of desired memory block to reallocate
 *
 * Returns:
 * SA_ALLOC_OVERFLOW(-1)	- invalid arguments, overflow
 * SA_ALLOC_SUCCESS	(0)		- memory reallocated successfully
 * SA_ALLOC_FAILURE	(1)		- out of memory
 */
enum sa_allocation_status am_aligned_realloc( void * out_buffer_pointer, void * pointer_original, size_t size_new );
/* Function:
 * safe version of am_aligned_realloc with array bounds checking
 *
 * Precondition:
 * pointer_original			- must be previously allocated with an am_aligned_malloc
 * elements_amount			- must be positive
 * element_size				- must be positive
 *
 * Parameters:
 * out_buffer_pointer		- holder to allocated memory
 * pointer_original			- existing aligned memory block
 * elements_amount			- amount of elements
 * element_size				- size of each element
 *
 * Returns:
 * SA_ALLOC_OVERFLOW(-1)	- invalid arguments, overflow
 * SA_ALLOC_SUCCESS	(0)		- memory reallocated successfully
 * SA_ALLOC_FAILURE	(1)		- out of memory
 */
enum sa_allocation_status am_aligned_realloc_array( void * out_buffer_pointer, void * pointer_original, size_t elements_amount, size_t element_size );
#		ifndef AM_NO_CALLOC
/* Function:
 * reallocate aligned memory block and zero out a specified tail portion
 *
 * Precondition:
 * out_buffer_pointer		- holder to allocated memory
 * pointer					- must be previously allocated with an am_aligned_malloc
 * size_new					- must be positive
 *
 * Parameters:
 * out_buffer_pointer		- holder to allocated memory
 * pointer					- existing aligned memory block
 * size_to_preserve			- size of memory block from the beginning to save
 * size_new					- new total size of memory block in bytes
 *
 * Returns:
 * SA_ALLOC_OVERFLOW(-1)	- invalid arguments, overflow
 * SA_ALLOC_SUCCESS	(0)		- memory reallocated successfully
 * SA_ALLOC_FAILURE	(1)		- out of memory
 */
enum sa_allocation_status am_aligned_recalloc( void * out_buffer_pointer, void * pointer, size_t size_to_preserve, size_t size_new );
/* Function:
 * safe version of am_aligned_recalloc with array bounds checking
 *
 * Precondition:
 * pointer					- must be previously allocated with an am_aligned_malloc
 * elements_amount			- must be positive
 * element_size				- must be positive
 *
 * Parameters:
 * out_buffer_pointer		- holder to allocated memory
 * pointer					- existing aligned memory block
 * elements_to_preserve		- amount of elements from the beginning to save
 * elements_amount			- amount of elements
 * element_size				- size of each element
 *
 * Returns:
 * SA_ALLOC_OVERFLOW(-1)	- invalid arguments, overflow
 * SA_ALLOC_SUCCESS	(0)		- memory reallocated successfully
 * SA_ALLOC_FAILURE	(1)		- out of memory
 */
enum sa_allocation_status am_aligned_recalloc_array( void * out_buffer_pointer, void * pointer, size_t elements_to_preserve, size_t elements_amount, size_t element_size );
#		endif /* AM_NO_CALLOC */
#	endif /* AM_NO_REALLOC */
#	ifndef AM_NO_CALLOC
/* Function:
 * allocate zero-initialized aligned memory for an array
 *
 * Parameters:
 * out_buffer_pointer	- holder to allocated memory
 * alignment			- desired address alignment
 * elements_amount		- amount of elements
 * element_size			- size of each element
 *
 * Returns:
 * true					- math is safe, aligned zero-filled memory allocation attempted
 *							(must be released with am_aligned_free)
 * false				- invalid arguments, overflow
 */
bool am_aligned_calloc( void * restrict out_buffer_pointer, size_t alignment, size_t elements_amount, size_t element_size );
#	endif /* AM_NO_CALLOC */

#endif/* ALIGNED_MEMORY_H */
