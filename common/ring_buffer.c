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

#include "ring_buffer.h"

#include "assert_m.h"				/* assert_check_m					*/
#include "safe_multiplication.h"	/* sa_ovf_mul_size_t				*/
#include "dynamic_array.h"			/* da_dynamic_array_ensure_capacity	*/

#include <string.h>	/* memcpy	*/

#include <stdbool.h>/* bool		*/
#include <stddef.h>	/* size_t	*/
#include <stdlib.h>	/* realloc	*/

static bool rb_ring_buffer_discard_internal(struct RB_Ring_Buffer * restrict ring_buffer_pointer);

bool rb_ring_buffer_create(
		struct RB_Ring_Buffer * restrict * restrict out_ring_buffer_pointer,
		size_t element_size, size_t capacity_desired, size_t capacity_minimal
	)
{
	if( assert_check_m( out_ring_buffer_pointer	!= NULL,"No buffer holder found"	) == false ||
		assert_check_m( element_size			!= 0,	"No buffer found"			) == false ||
		assert_check_mf(
			capacity_minimal <= capacity_desired,
			"Minimal capacity is larger than a desired one (%zu > %zu)",
			capacity_minimal, capacity_desired
		) == false ||
		assert_check_mf(
			sa_ovf_mul_size_t( capacity_minimal, element_size, &(size_t){0} ) == false,
			"Impossible buffer size (minimal capacity: %zu, size of element: %zu)",
			capacity_minimal, element_size
		) == false )
		return false;

	*out_ring_buffer_pointer = calloc( 1, sizeof(**out_ring_buffer_pointer) );
	if( *out_ring_buffer_pointer == NULL )
		return true;

	if (da_dynamic_array_ensure_capacity(
			&(*out_ring_buffer_pointer)->data_pointer, element_size,
			&(*out_ring_buffer_pointer)->capacity, capacity_desired, capacity_minimal
		) == false )
	{
		free( *out_ring_buffer_pointer );
		*out_ring_buffer_pointer = NULL;
		return true;
	}

	(*out_ring_buffer_pointer)->element_size = element_size;

	return true;
}

bool rb_ring_buffer_initialize_static(
		struct RB_Ring_Buffer * restrict ring_buffer_pointer, void * restrict buffer_pointer,
		size_t capacity, size_t element_size, size_t first, size_t amount, size_t first_free
	)
{
	if( assert_check_m( ring_buffer_pointer	!= NULL,"No ring buffer found"			) == false ||
		assert_check_m( buffer_pointer		!= NULL,"No buffer found"				) == false ||
		assert_check_m( capacity			!= 0,	"Buffer size required"			) == false ||
		assert_check_m( element_size		!= 0,	"Object size required"			) == false ||
		assert_check_m( amount <= capacity,			"Amount exceeds capacity"		) == false ||
		assert_check_m( first < capacity,			"First index exceeds capacity"	) == false ||
		assert_check_m( first_free < capacity,		"Free index exceeds capacity"	) == false ||
		assert_check_mf(
			sa_ovf_mul_size_t( capacity, element_size, &(size_t){0} ) == false,
			"Impossible buffer size (capacity: %zu, size of element: %zu)", capacity, element_size
		) == false )
		return false;

	bool is_state_valid;
	if ( first_free == first )
		is_state_valid = (amount == 0 || amount == capacity);
	else if ( first_free > first )
		is_state_valid = (first_free - first) == amount;
	else
		is_state_valid = (first - first_free) == (capacity - amount);

	if( assert_check_mf(
			is_state_valid == true,
			"Incorrect data (capacity: %zu, first: %zu, amount: %zu, first free: %zu)",
			capacity, first, amount, first_free
		) == false )
		return false;

	*ring_buffer_pointer = (struct RB_Ring_Buffer) {
		.data_pointer	= buffer_pointer,
		.capacity		= capacity,
		.first			= first,
		.amount			= amount,
		.first_free		= first_free,
		.element_size	= element_size
	};

	return true;
}

void rb_ring_buffer_destroy(struct RB_Ring_Buffer * restrict ring_buffer_pointer ) {
	assert_m( ring_buffer_pointer != NULL, "No ring buffer found to destroy" );
	if ( ring_buffer_pointer == NULL ) return;
	free( ring_buffer_pointer->data_pointer );
	free( ring_buffer_pointer );
}

void * rb_ring_buffer_pop( struct RB_Ring_Buffer * restrict ring_buffer_pointer ) {
	if( assert_check_m( ring_buffer_pointer != NULL, "No ring buffer found" ) == false ||
		ring_buffer_pointer->amount == 0 )
		return NULL;

	void * element_pointer = (char *)
		ring_buffer_pointer->data_pointer +
			ring_buffer_pointer->first * ring_buffer_pointer->element_size;

	ring_buffer_pointer->first = (ring_buffer_pointer->first + 1) % ring_buffer_pointer->capacity;
	--ring_buffer_pointer->amount;

	return element_pointer;
}

void * rb_ring_buffer_peek( const struct RB_Ring_Buffer * restrict ring_buffer_pointer ) {
	if( assert_check_m( ring_buffer_pointer != NULL, "No ring buffer found" ) == false ||
		ring_buffer_pointer->amount == 0 )
		return NULL;

	return (char *)
		ring_buffer_pointer->data_pointer +
			ring_buffer_pointer->first * ring_buffer_pointer->element_size;
}

void * rb_ring_buffer_peek_position(
		const struct RB_Ring_Buffer * restrict ring_buffer_pointer, size_t index
	)
{
	if( assert_check_m( ring_buffer_pointer != NULL, "No ring buffer found" ) == false ||
		ring_buffer_pointer->amount <= index )
		return NULL;

	size_t elements_at_right = ring_buffer_pointer->capacity - ring_buffer_pointer->first;

	if ( elements_at_right > index )
		return (void *) ((char *) ring_buffer_pointer->data_pointer +
			(ring_buffer_pointer->first + index) * ring_buffer_pointer->element_size );

	return (void *)
		( (char *) ring_buffer_pointer->data_pointer +
			(index - elements_at_right) * ring_buffer_pointer->element_size );
}

bool rb_ring_buffer_push(
		struct RB_Ring_Buffer * restrict ring_buffer_pointer, const void * restrict data_pointer
	)
{
	if( assert_check_m( ring_buffer_pointer	!= NULL, "No ring buffer found" ) == false ||
		assert_check_m( data_pointer		!= NULL, "No data to push found") == false ||
		ring_buffer_pointer->amount == ring_buffer_pointer->capacity )
		return false;

	char * destination_pointer = (char *)
		ring_buffer_pointer->data_pointer +
			ring_buffer_pointer->first_free * ring_buffer_pointer->element_size;

	memcpy( destination_pointer, data_pointer, ring_buffer_pointer->element_size );

	ring_buffer_pointer->first_free =
		(ring_buffer_pointer->first_free + 1) % ring_buffer_pointer->capacity;
	++ring_buffer_pointer->amount;

	return true;
}

bool rb_ring_buffer_shrink( struct RB_Ring_Buffer * restrict ring_buffer_pointer ) {
	if( assert_check_m( ring_buffer_pointer	!= NULL, "No ring buffer found" ) == false )
		return false;

	if ( ring_buffer_pointer->amount == 0 ) {
		if ( ring_buffer_pointer->data_pointer != NULL )
			free( ring_buffer_pointer->data_pointer );
		*ring_buffer_pointer = (struct RB_Ring_Buffer) {
			.element_size = ring_buffer_pointer->element_size
		};
		return true;
	}

	if ( ring_buffer_pointer->amount == ring_buffer_pointer->capacity )
		return true;

	bool is_wrapped = false;
	size_t data_to_move, first_position_original, first_original;
	char * data_pointer = (char *) ring_buffer_pointer->data_pointer;

	if ( ring_buffer_pointer->first_free < ring_buffer_pointer->first ) {
		is_wrapped = true;
		first_original = ring_buffer_pointer->first;
		first_position_original = ring_buffer_pointer->first * ring_buffer_pointer->element_size;
		data_to_move = (ring_buffer_pointer->capacity - ring_buffer_pointer->first) *
			ring_buffer_pointer->element_size;
		memmove(
			data_pointer + ring_buffer_pointer->first_free * ring_buffer_pointer->element_size,
			data_pointer + ring_buffer_pointer->first * ring_buffer_pointer->element_size,
			data_to_move
		);
		ring_buffer_pointer->first = ring_buffer_pointer->first_free;
	} else if ( ring_buffer_pointer->first != 0 ) {
		memmove(
			data_pointer,
			data_pointer + ring_buffer_pointer->first * ring_buffer_pointer->element_size,
			ring_buffer_pointer->amount * ring_buffer_pointer->element_size
		);
		ring_buffer_pointer->first = 0;
		ring_buffer_pointer->first_free = ring_buffer_pointer->amount;
	}

	void * temporary_pointer = realloc(
		data_pointer,
		ring_buffer_pointer->amount * ring_buffer_pointer->element_size
	);
	if ( temporary_pointer != NULL ) {
		ring_buffer_pointer->capacity = ring_buffer_pointer->amount;
		ring_buffer_pointer->first_free = ring_buffer_pointer->first;
		ring_buffer_pointer->data_pointer = temporary_pointer;
	} else {
		if ( is_wrapped == true ) {
			memmove(
				data_pointer + first_position_original,
				data_pointer + ring_buffer_pointer->first * ring_buffer_pointer->element_size,
				data_to_move
			);
			ring_buffer_pointer->first = first_original;
		}
		return false;
	}

	return true;
}

bool rb_ring_buffer_discard( struct RB_Ring_Buffer * restrict ring_buffer_pointer ) {
	if( assert_check_m( ring_buffer_pointer	!= NULL, "No ring buffer found" ) == false ||
		ring_buffer_pointer->amount == 0 )
		return false;

	return rb_ring_buffer_discard_internal( ring_buffer_pointer );	
}

bool rb_ring_buffer_is_full( const struct RB_Ring_Buffer * restrict ring_buffer_pointer ) {
	return
		assert_check_m(ring_buffer_pointer != NULL, "No ring buffer found") == true &&
		ring_buffer_pointer->amount == ring_buffer_pointer->capacity;
}

bool rb_ring_buffer_pop_copy(
		struct RB_Ring_Buffer * restrict ring_buffer_pointer, void * restrict data_pointer
	)
{
	if( assert_check_m( ring_buffer_pointer	!= NULL, "No ring buffer found" ) == false ||
		assert_check_m( data_pointer		!= NULL, "No data holder found" ) == false)
		return false;

	const void * source_data_pointer = rb_ring_buffer_peek( ring_buffer_pointer );
	if ( source_data_pointer == NULL )
		return false;

	memcpy( data_pointer, source_data_pointer, ring_buffer_pointer->element_size );

	return rb_ring_buffer_discard_internal( ring_buffer_pointer );
}

bool rb_ring_buffer_ensure_capacity(
		struct RB_Ring_Buffer * restrict ring_buffer_pointer, size_t capacity_desired
	)
{
	if( assert_check_m( ring_buffer_pointer	!= NULL, "No ring buffer found" ) == false )
		return false;

	if ( ring_buffer_pointer->capacity >= capacity_desired )
		return true;

	size_t capacity_old = ring_buffer_pointer->capacity;
	size_t capacity_base = ring_buffer_pointer->capacity ? ring_buffer_pointer->capacity : 8;
	if ( da_dynamic_array_ensure_capacity(
			&ring_buffer_pointer->data_pointer, ring_buffer_pointer->element_size,
			&ring_buffer_pointer->capacity, capacity_desired, capacity_base
		) == false )
		return false;

	if( ring_buffer_pointer->first > ring_buffer_pointer->first_free ||
		(ring_buffer_pointer->first == ring_buffer_pointer->first_free &&
			ring_buffer_pointer->amount != 0) )
	{
		size_t elements_to_move = capacity_old - ring_buffer_pointer->first;
		size_t element_new_position = ring_buffer_pointer->capacity - elements_to_move;
		memmove(
			(char *) ring_buffer_pointer->data_pointer +
				element_new_position * ring_buffer_pointer->element_size,
			(char *) ring_buffer_pointer->data_pointer + ring_buffer_pointer->first *
				ring_buffer_pointer->element_size,
			elements_to_move * ring_buffer_pointer->element_size
		);
		ring_buffer_pointer->first = element_new_position;
	}

	return true;
}

size_t rb_ring_buffer_amount_free( const struct RB_Ring_Buffer * restrict ring_buffer_pointer ) {
	assert_m( ring_buffer_pointer != NULL, "No ring buffer found" );
	return ring_buffer_pointer->capacity - ring_buffer_pointer->amount;
}

/* Function:
 * internal discard that assume ring_buffer_pointer is correct
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Returns:
 * true					- element removed
 */
static bool rb_ring_buffer_discard_internal( struct RB_Ring_Buffer * restrict ring_buffer_pointer )
{
	ring_buffer_pointer->first = (ring_buffer_pointer->first + 1) % ring_buffer_pointer->capacity;
	--ring_buffer_pointer->amount;

	return true;
}
