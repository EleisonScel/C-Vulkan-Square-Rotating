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

#ifndef SAFE_ALLOC_H
#define SAFE_ALLOC_H

enum sa_allocation_status {
	SA_ALLOC_OVERFLOW	= -1,
	SA_ALLOC_FAILURE	= 1,
	SA_ALLOC_SUCCESS	= 0
};

#	include "assert_m.h"			/* assert_check_m	*/
#	include "safe_multiplication.h"	/* sa_ovf_mul_size_t*/

#	include <string.h> /* memset	*/

#	include <stddef.h> /* size_t	*/
#	include <stdlib.h> /* malloc	*/
#	include <stdbool.h>/* bool		*/

static inline size_t sa_check_array_bounds( size_t elements_amount, size_t element_size );

/* Function:
 * safe version of malloc with array bounds checking
 *
 * out_buffer_pointer	- holder to allocated memory
 * elements_amount		- desired amount of elements
 * element_size			- size of each element
 *
 * Returns:
 * true					- math is safe, memory allocation attempted
 * false				- invalid arguments, overflow
 */
static inline bool sa_malloc_array(
		void * restrict out_buffer_pointer, size_t elements_amount, size_t element_size
	)
{
	size_t total_bytes = sa_check_array_bounds( elements_amount, element_size );

	if (total_bytes == 0) {
		void * null_pointer = NULL;
		memcpy( out_buffer_pointer, &null_pointer, sizeof(void*) );
		return false;
	}

	void * pointer_new = malloc(total_bytes);
	memcpy( out_buffer_pointer, &pointer_new, sizeof(void*) );

	return true;
}

/* Function:
 * safe version of calloc with array bounds checking
 *
 * out_buffer_pointer	- holder to allocated memory
 * elements_amount		- desired amount of elements
 * element_size			- size of each element
 *
 * Returns:
 * true					- math is safe, memory allocation attempted
 * false				- invalid arguments, overflow
 */
static inline bool sa_calloc_array(
		void * restrict out_buffer_pointer, size_t elements_amount, size_t element_size
	)
{
	if ( sa_check_array_bounds( elements_amount, element_size ) == 0 ) {
		void * null_pointer = NULL;
		memcpy( out_buffer_pointer, &null_pointer, sizeof(void*) );
		return false;
	}

	void * pointer_new = calloc( elements_amount, element_size );
	memcpy( out_buffer_pointer, &pointer_new, sizeof(void*) );

	return true;
}

/* Function:
 * safe version of realloc with array bounds checking
 *
 * out_buffer_pointer		- holder to allocated memory
 * pointer_original			- original memory pointer
 * elements_amount			- desired amount of elements
 * element_size				- size of each element
 *
 * Returns:
 * SA_ALLOC_OVERFLOW(-1)	- invalid arguments, overflow
 * SA_ALLOC_SUCCESS	(0)		- memory reallocated successfully
 * SA_ALLOC_FAILURE	(1)		- out of memory
 */
static inline enum sa_allocation_status sa_realloc_array(
		void * out_buffer_pointer, void * pointer_original, size_t elements_amount,
		size_t element_size
	)
{
	size_t total_bytes = sa_check_array_bounds( elements_amount, element_size );

	if (total_bytes == 0)
		return SA_ALLOC_OVERFLOW;

	void * pointer_new = realloc( pointer_original, total_bytes );
	if ( pointer_new == NULL )
		return SA_ALLOC_FAILURE;
	memcpy( out_buffer_pointer, &pointer_new, sizeof(void*) );

	return SA_ALLOC_SUCCESS;
}

/* Function:
 * safe version of realloc with nullification of a new data with array bounds checking
 *
 * out_buffer_pointer		- holder to allocated memory
 * pointer_original			- original memory pointer
 * elements_amount_old		- old amount of elements
 * elements_amount_new		- desired amount of elements
 * element_size				- size of each element
 *
 * Returns:
 * SA_ALLOC_OVERFLOW(-1)	- invalid arguments, overflow
 * SA_ALLOC_SUCCESS	(0)		- memory reallocated successfully
 * SA_ALLOC_FAILURE	(1)		- out of memory
 */
static inline enum sa_allocation_status sa_recalloc_array(
		void * out_buffer_pointer, void * pointer_original, size_t elements_amount_old,
		size_t elements_amount_new, size_t element_size
	)
{
	size_t total_bytes = sa_check_array_bounds( elements_amount_new, element_size );

	if (total_bytes == 0)
		return SA_ALLOC_OVERFLOW;

	void * pointer_new = realloc( pointer_original, total_bytes );
	if ( pointer_new == NULL )
		return SA_ALLOC_FAILURE;

	memcpy( out_buffer_pointer, &pointer_new, sizeof(void*) );

	if ( elements_amount_new > elements_amount_old ) {
		char * pointer_to_clean_from = (char *) pointer_new + (elements_amount_old * element_size);
		memset(
			pointer_to_clean_from,
			0,
			(elements_amount_new - elements_amount_old) * element_size
		);
	}

	return SA_ALLOC_SUCCESS;
}

/* Function:
 * validate the allocation size
 *
 * elements_amount	- amount of elements
 * element_size		- size of each element
 *
 * Returns:
 * non-zero			- allocation size
 * zero				- incorrect arguments
 */
static inline size_t sa_check_array_bounds( size_t elements_amount, size_t element_size ) {
	size_t total_bytes;
	if( assert_check_m(elements_amount != 0, "Amount of elements for allocation shouldn't be zero")
		== false ||
		assert_check_m(element_size	 != 0, "Element size shouldn't be zero, it's a divider")
		== false || sa_ovf_mul_size_t(elements_amount, element_size, &total_bytes ) == true )
		return 0;

	return total_bytes;
}

#endif /* SAFE_ALLOC_H */
