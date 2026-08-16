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

#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#	include <stddef.h>	/* size_t	*/
#	include <stdbool.h>	/* bool		*/

struct RB_Ring_Buffer {
	void	* data_pointer;	/* ring buffer				*/
	size_t	capacity;		/* size of the ring buffer	*/
	size_t	first;			/* first valid element		*/
	size_t	amount;			/* number of elements		*/
	size_t	first_free;		/* first free position		*/
	size_t	element_size;	/* size of single element	*/
};

/* functions UNIVERSAL */

/* Function:
 * pop an element out of the ring buffer
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 *
 * Returns:
 * pointer				- element popped out successfully
 * NULL					- invalid arguments, empty buffer
 */
void * rb_ring_buffer_pop( struct RB_Ring_Buffer * restrict ring_buffer_pointer );
/* Function:
 * peek at the first element of the ring buffer without removing it
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 *
 * Returns:
 * pointer				- pointer of the first element gotten successfully
 * NULL					- invalid arguments
 */
void * rb_ring_buffer_peek( const struct RB_Ring_Buffer * restrict ring_buffer_pointer );
/* Function:
 * peek at the index element of the ring buffer without removing it
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 * index				- logical index of the element [0; amount of elements)
 *
 * Returns:
 * pointer				- pointer to the element at the given index
 * NULL					- invalid arguments, index out of bounds
 */
void * rb_ring_buffer_peek_position( const struct RB_Ring_Buffer * restrict ring_buffer_pointer, size_t index );

/* Function:
 * push an element to the ring buffer
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 * data_pointer			- pointer to the data to push
 *
 * Returns:
 * true					- element pushed successfully
 * false				- invalid arguments
 */
bool rb_ring_buffer_push( struct RB_Ring_Buffer * restrict ring_buffer_pointer, const void * restrict data_pointer );
/* Function:
 * discard the first element from the ring buffer
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 *
 * Returns:
 * true					- element discarded successfully
 * false				- invalid arguments, buffer is empty
 */
bool rb_ring_buffer_discard( struct RB_Ring_Buffer * restrict ring_buffer_pointer );
/* Function:
 * check if the ring buffer is full
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 *
 * Returns:
 * true					- buffer is full
 * false				- invalid arguments, buffer is not full
 */
bool rb_ring_buffer_is_full( const struct RB_Ring_Buffer * restrict ring_buffer_pointer );
/* Function:
 * pop out first element from the ring buffer and copy it to a destination
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 * data_pointer			- destination to copy the popped element to
 *
 * Returns:
 * true					- element popped and copied
 * false				- invalid arguments, empty buffer
 */
bool rb_ring_buffer_pop_copy( struct RB_Ring_Buffer * restrict ring_buffer_pointer, void * restrict data_pointer );
/* Function:
 * initialize a ring buffer with an external memory block (static or dynamic)
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 * buffer_pointer		- isn't NULL
 * capacity				- isn't 0, doesn't cause an overflow by multiplying with element_size
 * element_size			- isn't 0, doesn't cause an overflow by multiplying with capacity
 * amount				- less or equal to capacity
 * first				- less than capacity
 * first_free			- less than capacity
 * state of the buffer	-
 	1. if last element is also the first element amount must be equal capacity or 0
 	2. if last element is after first one their difference must be equal to amount
 	3. if last element is before first one their difference by modulo must be equal
 		to the free space
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 * buffer_pointer		- external memory block use as storage
 * capacity				- capacity of the ring buffer
 * element_size			- size of the element
 * first				- index of first valid element
 * amount				- number of elements in the buffer
 * first_free			- first free element position
 *
 * Returns:
 * true					- ring buffer initialized successfully
 * false				- invalid arguments, overflow, incorrect state
 */
bool rb_ring_buffer_initialize_static( struct RB_Ring_Buffer * ring_buffer_pointer, void * restrict buffer_pointer, size_t capacity, size_t element_size, size_t first, size_t amount, size_t first_free );

/* Function:
 * get the amount of free slots in the ring buffer
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 *
 * Returns:
 * value				- number of free slots
 */
size_t rb_ring_buffer_amount_free( const struct RB_Ring_Buffer * restrict ring_buffer_pointer );

/* functions DYNAMIC */
/* Function:
 * destroy a dynamic ring buffer and free its memory
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 */
void rb_ring_buffer_destroy(struct RB_Ring_Buffer * restrict ring_buffer_pointer );

/* Function:
 * shrink the ring buffer capacity to the exact amount of elements
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 *
 * Returns:
 * true					- no need to shrink or buffer shrinked successfully
 * false				- invalid arguments, memory allocation failed
 */
bool rb_ring_buffer_shrink( struct RB_Ring_Buffer * restrict ring_buffer_pointer );
/* Function:
 * create a dynamic ring buffer
 *
 * Precondition:
 * out_ring_buffer_pointer	- isn't NULL
 * element_size				- isn't 0
 * capacity_minimal			- less or equal to capacity_desired
 * capacity_desired			- more or equal to capacity_minimal
 *
 * Parameters:
 * out_ring_buffer_pointer	- ring buffer pointer holder
 * element_size				- size of the element
 * capacity_desired			- desired capacity of the ring buffer
 * capacity_minimal			- minimal capacity of the ring buffer
 *
 * Returns:
 * true						- ring buffer initialized successfully
 * false					- invalid arguments, overflow, incorrect state
 */
bool rb_ring_buffer_create( struct RB_Ring_Buffer * restrict * restrict out_ring_buffer_pointer, size_t element_size, size_t capacity_desired, size_t capacity_minimal );
/* Function:
 * ensure the ring buffer has at least the specific capacity
 *
 * Precondition:
 * ring_buffer_pointer	- isn't NULL
 *
 * Parameters:
 * ring_buffer_pointer	- ring buffer pointer
 * capacity_desired		- minimal desired capacity
 *
 * Returns:
 * true					- capacity is sufficient or expanded successfully
 * false				- invalid arguments, memory allocation failed
 */
bool rb_ring_buffer_ensure_capacity( struct RB_Ring_Buffer * restrict ring_buffer_pointer, size_t capacity_desired );

#endif /* RING_BUFFER_H */
