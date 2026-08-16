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

#include "write_out_error_message.h"

#include "dynamic_array.h"	/* da_dynamic_array_shrink	*/
#include "assert_m.h"		/* assert_m					*/

#include <stdio.h>	/* vsnprintf*/
#include <stdarg.h>	/* va_list	*/
#include <stddef.h>	/* size_t	*/
#include <stdlib.h>	/* malloc	*/
#include <stdbool.h>/* bool		*/

#define WOEM_STATIC_CAPACITY	8
#define WOEM_BASE_ERRORS_AMOUNT	4

static char woem_error_malloc_failed[] =
	"(write_out_error_message) memory allocation for error text failed";
static char woem_error_no_message	[] = "(write_out_error_message) no error message specified";
static char woem_error_formatting	[] = "(write_out_error_message) formatting error";
static char woem_error_encoding		[] = "(write_out_error_message) encoding error";

struct WOEM_Error_Unit {
	char * error_message_pointer;
	bool must_be_freed;
};

static struct WOEM_Error_Handle {
	struct WOEM_Error_Unit		static_array[WOEM_STATIC_CAPACITY];
	struct WOEM_Error_Unit		* dynamic_array;
	size_t						dynamic_capacity;
	size_t						amount;
} current_error_handle = { 0 };

static bool woem_write_out_error_message( char ** restrict out_error_message, const char * restrict format_pointer, va_list arguments );
static bool woem_store_error_message( char * restrict error_message_pointer, bool must_be_freed );
static bool woem_ensure_capacity( void );

bool woem_push( const char * restrict format_pointer, ... ) {
	if( assert_check_m( format_pointer != NULL, "No error message found" ) == false )
		return false;

	bool message_must_be_freed = false;
	char * error_message_pointer = NULL;

	va_list arguments;
	va_start( arguments, format_pointer );
	message_must_be_freed =
		woem_write_out_error_message(&error_message_pointer, format_pointer, arguments);
	va_end( arguments );
	assert_m( error_message_pointer != NULL, "Error message failed to get" );

	return woem_store_error_message( error_message_pointer, message_must_be_freed );
}

bool woem_push_raw( char * restrict error_message_pointer ) {
	if( assert_check_m( error_message_pointer != NULL, "No error message found" ) == false )
		return false;

	return woem_store_error_message( error_message_pointer, true );
}

void woem_clear(void) {
	/* clean the dynamic error messages */
	if ( current_error_handle.dynamic_array != NULL ) {
		assert_m(
			current_error_handle.amount > WOEM_STATIC_CAPACITY,
			"Dynamic error messages shouldn't exist when static space suffices"
		);
		for(size_t index = current_error_handle.amount > WOEM_STATIC_CAPACITY
				? current_error_handle.amount - WOEM_STATIC_CAPACITY
				: 0;
			index > 0; --index )
		{
			if ( current_error_handle.dynamic_array		[index - 1].must_be_freed )
				free( current_error_handle.dynamic_array[index - 1].error_message_pointer );
		}
		free( current_error_handle.dynamic_array );
		current_error_handle.dynamic_capacity	= 0;
		current_error_handle.dynamic_array		= NULL;
		current_error_handle.amount				= WOEM_STATIC_CAPACITY;
	}

	/* clean the static error messages */
	for ( size_t index = 0; index < current_error_handle.amount; ++index ) {
		if ( current_error_handle.static_array[index].must_be_freed )
			free( current_error_handle.static_array[index].error_message_pointer );
	}

	current_error_handle.amount = 0;
}

void woem_shrink(void) {
	assert_m(
		(current_error_handle.dynamic_array == NULL) ==
		(current_error_handle.amount <= WOEM_STATIC_CAPACITY),
		"Static array shall be filled first"
	);

	if (current_error_handle.dynamic_array == NULL ||
		current_error_handle.amount <= WOEM_STATIC_CAPACITY )
		return;

	/* da_dynamic_array_shrink return false only on invalid arguments,			*
	 * which is a programmer error. If realloc failed to reduce memory block,	*
	 * it keeps the original memory untouched and returns true;					*
	 * This return value may be safely ignored.									*
	 */
	bool result = da_dynamic_array_shrink(
		&current_error_handle.dynamic_array,
		sizeof( *current_error_handle.dynamic_array ),
		&current_error_handle.dynamic_capacity,
		current_error_handle.amount - WOEM_STATIC_CAPACITY,
		WOEM_BASE_ERRORS_AMOUNT
	);
	assert_m( result, "Incorrect arguments to shrink and it failed" );
	(void) result;
}

char * woem_pop( bool * restrict out_must_be_freed_pointer ) {
	if( assert_check_m( out_must_be_freed_pointer != NULL, "No free flag found" ) == false )
		return NULL;
	if ( current_error_handle.amount == 0 ) {
		*out_must_be_freed_pointer = false;
		return NULL;
	}

	struct WOEM_Error_Unit current_unit = { 0 };
	if ( current_error_handle.dynamic_array == NULL )
		current_unit = current_error_handle.static_array
			[current_error_handle.amount - 1];
	else
		current_unit = current_error_handle.dynamic_array
			[current_error_handle.amount - WOEM_STATIC_CAPACITY - 1];

	--current_error_handle.amount;
	if (current_error_handle.amount <= WOEM_STATIC_CAPACITY &&
		current_error_handle.dynamic_array != NULL )
	{
		free( current_error_handle.dynamic_array );
		current_error_handle.dynamic_array		= NULL;
		current_error_handle.dynamic_capacity	= 0;
	}

	*out_must_be_freed_pointer = current_unit.must_be_freed;

	return current_unit.error_message_pointer;
}

/* Function:
 * take ownership of the string and save it
 *
 * Parameters:
 * error_message_pointer	- pointer to an error message
 * must_be_freed			- flag signaling if it's a heap or stack memory
 *
 * Returns:
 * true						- error message saved
 * false					- out of memory (if the string is heap-allocated, it's freed)
 */
static bool woem_store_error_message( char * restrict error_message_pointer, bool must_be_freed ) {
	assert_m( error_message_pointer != NULL, "No error message found" );
	if ( current_error_handle.amount < WOEM_STATIC_CAPACITY ) {
		current_error_handle.static_array[current_error_handle.amount].error_message_pointer =
			error_message_pointer;
		current_error_handle.static_array[current_error_handle.amount].must_be_freed =
			must_be_freed;
		goto out;
	}

	if ( woem_ensure_capacity() == false ) {
		if ( must_be_freed == true ) free( error_message_pointer );
		return false;
	}

	current_error_handle.dynamic_array[current_error_handle.amount - WOEM_STATIC_CAPACITY]
		.error_message_pointer = error_message_pointer;
	current_error_handle.dynamic_array[current_error_handle.amount - WOEM_STATIC_CAPACITY]
		.must_be_freed = must_be_freed;
out:
	++current_error_handle.amount;
	return true;
}

/* Function:
 * format error into a heap memory if possible
 *
 * Parameters:
 * out_error_message	- pointer to write error message in
 * format_pointer		- format string
 * arguments			- format arguments
 *
 * Returns:
 * true					- the error message must be freed
 * false				- writing error message failed; this is a static diagnostic message
 */
static bool woem_write_out_error_message(
		char ** restrict out_error_message,
		const char * restrict format_pointer, va_list arguments
	)
{
	char * buffer_pointer = NULL;
	/* int required by vsnprintf */
	int written = 0, chars_needed = 0;

	if( assert_check_m( out_error_message != NULL, "No error message handle found" ) == false )
		goto cleanup;
	*out_error_message = NULL;

	if ( assert_check_m( format_pointer != NULL, "No error message specified" ) == false ) {
		*out_error_message = woem_error_no_message;
		goto cleanup;
	}

	va_list arguments_temporary;
	va_copy( arguments_temporary, arguments );
	chars_needed = vsnprintf( NULL, 0, format_pointer, arguments_temporary );
	va_end( arguments_temporary );

	if ( chars_needed < 0 ) {
		*out_error_message = woem_error_encoding;
		goto cleanup;
	}

#	if INT_MAX >= SIZE_MAX

	if ( (unsigned int) chars_needed + 1 > (unsigned int) SIZE_MAX ) {
		*out_error_message = woem_error_malloc_failed;
		goto cleanup;
	}

#	endif /* INT_MAX >= SIZE_MAX */

	buffer_pointer = malloc( (size_t) chars_needed + 1 );
	if ( buffer_pointer == NULL ) {
		*out_error_message = woem_error_malloc_failed;
		goto cleanup;
	}

	written = vsnprintf(
		buffer_pointer, (size_t)chars_needed + 1, format_pointer, arguments
	);
	if ( written != chars_needed ) {
		*out_error_message = woem_error_formatting;
		goto cleanup;
	}

	*out_error_message = buffer_pointer;
	return true;

cleanup:
	if ( buffer_pointer ) free( buffer_pointer );
	return false;
}

/* Function:
 * guarantee dynamic array space for at least one more error
 * (shall only be called when the static array is full)
 *
 * Returns:
 * true		- space is available
 * false	- allocation space failed
 */
static bool woem_ensure_capacity(void) {
	assert_m(
		current_error_handle.amount >= WOEM_STATIC_CAPACITY,
		"To use a dynamic buffer the static shall be full"
	);
	if ( current_error_handle.dynamic_capacity
			>= current_error_handle.amount - WOEM_STATIC_CAPACITY + 1 )
		return true;

	return da_dynamic_array_ensure_capacity(
		&current_error_handle.dynamic_array, sizeof(*current_error_handle.dynamic_array),
		&current_error_handle.dynamic_capacity,
		current_error_handle.amount - WOEM_STATIC_CAPACITY + 1,
		WOEM_BASE_ERRORS_AMOUNT
	);
}
