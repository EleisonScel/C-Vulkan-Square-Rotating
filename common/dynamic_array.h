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

#ifndef DA_DYNAMIC_ARRAY_H
#define DA_DYNAMIC_ARRAY_H

#	include <stddef.h>	/* size_t	*/
#	include <stdbool.h>	/* bool		*/

struct DA_Dynamic_Array_List {
	void	* data_pointer;
	size_t	item_size;
};

/* Function:
 * free and nullify dynamic array
 *
 * Parameters:
 * data_pointer		- data pointer to free
 * amount_pointer	- amount to nullify
 * capacity_pointer	- capacity to nullify
 */
void da_dynamic_array_free( void * restrict data_pointer, size_t * restrict amount_pointer, size_t * restrict capacity_pointer );

/* Function:
 * shrink dynamic array to a given size
 *
 * Parameters:
 * data_pointer		- array
 * item_size		- size of the array element
 * capacity_pointer	- pointer to the capacity of array
 * amount			- desired amount of array elements
 * base_amount		- minimal array size
 *
 * Returns:
 * false			- invalid arguments
 * true				- shrinking isn't needed or it's completed
 */
bool da_dynamic_array_shrink( void * restrict data_pointer, size_t item_size, size_t * restrict capacity_pointer, size_t amount, size_t base_amount );
/* Function:
 * guarantees dynamic array to be at least needed size
 *
 * Parameters:
 * data_pointer		- array
 * item_size		- size of the array element
 * capacity_pointer	- capacity of array
 * needed			- needed amount
 * base_amount		- base array size (first call only)
 *
 * Returns:
 * false			- dynamic array extension failed
 * true				- dynamic array extended
 */
bool da_dynamic_array_ensure_capacity( void * restrict data_pointer, size_t item_size, size_t * restrict capacity_pointer, size_t needed, size_t base_amount );
/* Function:
 * guarantees list of dynamic arrays to be at least needed size
 *
 * Parameters:
 * list				- list of dynamic arrays
 * list_amount		- size of list of dynamic arrays
 * capacity_pointer	- capacity of arrays
 * needed			- needed amount of elements in an array
 * base_amount		- base array size (first call only)
 *
 * Returns:
 * false			- list of dynamic arrays extension failed
 * true				- list of dynamic arrays extended
 */
bool da_dynamic_array_ensure_capacity_list( const struct DA_Dynamic_Array_List * restrict list_pointer, size_t list_amount, size_t * restrict capacity_pointer, size_t needed, size_t base_amount );

#endif /* DA_DYNAMIC_ARRAY_H */
