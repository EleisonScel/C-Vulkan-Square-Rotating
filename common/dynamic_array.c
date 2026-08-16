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

#include "dynamic_array.h"

#include "clamp_values.h"	/* cv_clamp_size_t	*/
#include "assert_m.h"		/* assert_m			*/

#include <stdint.h>		/* SIZE_MAX	*/
#include <stddef.h>		/* size_t	*/
#include <stdlib.h>		/* malloc	*/
#include <stdbool.h>	/* bool		*/

#include <string.h>		/* memcpy	*/

static bool da_dynamic_array_expand_list( const struct DA_Dynamic_Array_List * restrict
	list_pointer, size_t list_amount, size_t capacity_old, size_t capacity_target );
static void da_dynamic_array_restore_all( const struct DA_Dynamic_Array_List * restrict
	list_pointer, size_t list_amount, size_t capacity_target );

bool da_dynamic_array_shrink(
		void * restrict data_pointer, size_t item_size, size_t * restrict capacity_pointer,
		size_t amount, size_t base_amount
	)
{
	if( assert_check_m( item_size		!= 0,	"Element size can not be zero"	) == false ||
		assert_check_m( data_pointer	!= NULL,"No array found"				) == false ||
		assert_check_m( capacity_pointer!= NULL,"Capacity isn't found"			) == false)
		return false;

	if ( amount >= *capacity_pointer || base_amount >= *capacity_pointer )
		return true;

	if( amount < base_amount )
		amount = base_amount;

	void * current_data_pointer = NULL;
	memcpy(
		&current_data_pointer,
		data_pointer,
		sizeof( data_pointer )
	);
	if ( amount == 0 ) {
		free( current_data_pointer );
		current_data_pointer	= NULL;
		memcpy(
			data_pointer,
			&current_data_pointer,
			sizeof( current_data_pointer )
		);
		*capacity_pointer		= amount;
		return true;
	}

	void * new_pointer = realloc( current_data_pointer, amount * item_size );
	if ( new_pointer != NULL ) {
		memcpy(
			data_pointer,
			&new_pointer,
			sizeof( new_pointer )
		);
		*capacity_pointer	= amount;
	}

	return true;
}

bool da_dynamic_array_ensure_capacity(
		void * restrict data_pointer, size_t item_size, size_t * restrict capacity_pointer,
		size_t needed, size_t base_amount
	)
{
	if( assert_check_m( item_size != 0, "Element size can not be zero"	) == false ||
		assert_check_m( data_pointer != NULL, "No array found"			) == false ||
		assert_check_m( capacity_pointer != NULL, "Capacity isn't found") == false ||
		assert_check_m(*capacity_pointer > 0 || base_amount > 0,
			"Base amount and capacity are both zero"					) == false)
		return false;

	if( needed <= *capacity_pointer )
		return true;

	const size_t capacity_maximal = SIZE_MAX / item_size;
	if ( needed > capacity_maximal )
		return false;

	const size_t capacity_target = (*capacity_pointer != 0)
		? *capacity_pointer + (*capacity_pointer >> 1)
		: base_amount;
	size_t capacity_new = cv_clamp_size_t(
		capacity_target,
		needed,
		capacity_maximal
	);

	void * current_data_pointer = NULL;
	memcpy(
			&current_data_pointer,
			data_pointer,
			sizeof( current_data_pointer )
	);

	void * new_pointer = realloc( current_data_pointer, capacity_new * item_size );
	if ( new_pointer == NULL ) {
		if ( capacity_new != needed ) {
			capacity_new = needed;
			/* overflow check is omitted here because capacity_new <= capacity_maximal */
			new_pointer = realloc( current_data_pointer, capacity_new * item_size );
			if ( new_pointer == NULL ) return false;
		} else return false;
	}

	memcpy(
			data_pointer,
			&new_pointer,
			sizeof( new_pointer )
	);
	*capacity_pointer	= capacity_new;

	return true;
}

bool da_dynamic_array_ensure_capacity_list(
		const struct DA_Dynamic_Array_List * restrict list_pointer, size_t list_amount,
		size_t * restrict capacity_pointer, size_t needed, size_t base_amount
	)
{
	if( assert_check_m( list_pointer	!= NULL, "No arrays found" 					) == false ||
		assert_check_m( capacity_pointer!= NULL, "Capacity isn't found"				) == false ||
		assert_check_m( list_amount > 0, "Arrays to check shall be positive amount"	) == false ||
		assert_check_m( *capacity_pointer > 0 || base_amount > 0,
			"Base amount and capacity are both zero"								) == false)
		return false;

	if ( needed <= *capacity_pointer )
		return true;

	size_t capacity_maximal = SIZE_MAX;
	for ( size_t capacity_check_i = 0; capacity_check_i < list_amount; ++capacity_check_i )
	{
		if( assert_check_m(
				list_pointer[capacity_check_i].item_size != 0, "item size shall not be 0"
			) == false )
			return false;

		const size_t capacity_maximal_current =
			SIZE_MAX / list_pointer[capacity_check_i].item_size;
		if ( needed > capacity_maximal_current )
			return false;
		if ( capacity_maximal_current < capacity_maximal )
			capacity_maximal = capacity_maximal_current;
	}

	const size_t capacity_target = (*capacity_pointer != 0)
		? *capacity_pointer + (*capacity_pointer >> 1)
		: base_amount;

	size_t capacity_new[2] = {
		cv_clamp_size_t(capacity_target, needed, capacity_maximal), needed
	};
	size_t tries = (capacity_new[0] == capacity_new[1]) ? 1 : 2;
	for(size_t try_index = 0; try_index < tries; ++try_index )
	{
		if (da_dynamic_array_expand_list(
				list_pointer, list_amount, *capacity_pointer, capacity_new[try_index] )
			)
		{
			*capacity_pointer = capacity_new[try_index];
			return true;
		}
	}
	return false;
}

void da_dynamic_array_free(
		void * restrict data_pointer, size_t * restrict amount_pointer,
		size_t * restrict capacity_pointer
	)
{
	assert_m( capacity_pointer	!= NULL, "Capacity isn't found"					);
	assert_m( amount_pointer	!= NULL, "Amount of arrays to free isn't found"	);
	assert_m( data_pointer		!= NULL, "No array found to free"				);

	if ( data_pointer != NULL ) {
		void * current_data_pointer = NULL;
		memcpy(
				&current_data_pointer,
				data_pointer,
				sizeof( current_data_pointer )
		);
		free( current_data_pointer );
		current_data_pointer = NULL;
		memcpy(
				data_pointer,
				&current_data_pointer,
				sizeof( current_data_pointer )
		);
	}
	if ( amount_pointer != NULL )
		*amount_pointer = 0;
	if ( capacity_pointer != NULL )
		*capacity_pointer = 0;
}

/* Function:
 * expand the list of dynamic array
 *
 * Parameters:
 * list_pointer		- list of dynamic arrays
 * list_amount		- size of list of dynamic arrays
 * capacity_old		- old capacity of arrays
 * capacity_target	- desired capacity of arrays
 *
 * Returns:
 * false			- list of dynamic arrays extension failed
 * true				- list of dynamic arrays extended
 */
static bool da_dynamic_array_expand_list(
		const struct DA_Dynamic_Array_List * restrict list_pointer, size_t list_amount,
		size_t capacity_old, size_t capacity_target
	)
{
	assert_m( list_amount		> 0,			"Arrays to check shall be positive amount"	);
	assert_m( list_pointer		!= NULL,		"No arrays found"							);
	assert_m( capacity_target	> capacity_old,	"Desired capacity isn't greater than old"	);

	for( size_t size_check_index = 0; size_check_index < list_amount; ++size_check_index ) {
		assert_m( list_pointer[size_check_index].item_size != 0, "Item size shall not be 0" );
		if ( capacity_target > SIZE_MAX / list_pointer[size_check_index].item_size )
			return false;
	}

	for( size_t list_index = 0; list_index < list_amount; ++list_index ) {
		void ** data_pointer = (void **) list_pointer[list_index].data_pointer;
		assert_m( list_pointer[list_index].item_size != 0, "Item size shall not be 0" );
		void * temporary_pointer =
			realloc( *data_pointer, capacity_target * list_pointer[list_index].item_size );
		if ( temporary_pointer == NULL )
		{
			da_dynamic_array_restore_all( list_pointer, list_index, capacity_old );
			return false;
		}
		*data_pointer = temporary_pointer;
	}
	return true;
}

/* Function:
 * restore list of dynamic arrays till the desired capacity size
 *
 * Parameters:
 * list_pointer		- list of dynamic arrays
 * list_amount		- size of list of dynamic arrays
 * capacity_target	- desired capacity of arrays
 */
static void da_dynamic_array_restore_all(
		const struct DA_Dynamic_Array_List * restrict list_pointer, size_t list_amount,
		size_t capacity_target
	)
{
	assert_m( list_pointer	!= NULL,	"No arrays found"							);
	assert_m( list_amount	> 0,		"Arrays to check shall be positive amount"	);

	for ( size_t list_index = 0; list_index < list_amount; ++list_index ) {
		void ** data_pointer = (void **)list_pointer[list_index].data_pointer;
		if ( capacity_target != 0 ) {
			assert_m( list_pointer[list_index].item_size != 0, "Item size shall not be 0" );
			void * temporary_pointer =
				realloc( *data_pointer, capacity_target * list_pointer[list_index].item_size );
			if ( temporary_pointer != NULL ) *data_pointer = temporary_pointer;
		} else {
			free( *data_pointer );
			*data_pointer = NULL;
		}
	}
}
