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

#include "cleanup_register.h"

#include "assert_m.h"	/* assert_m */

#include <stdlib.h>		/* atexit	*/
#include <stdbool.h>	/* bool		*/

static struct CR_Cleanup_Handle {
	void (* function_pointer)( void * );
	void * argument_pointer;
} cr_current_handle = { 0 };

static bool cr_registered_atexit = false;

static void cr_cleanup_wrapper(void);

void cr_register_cleanup_wrapper(
		void (*function_pointer)(void *), void * restrict argument_pointer
	)
{
	assert_m(
		function_pointer != NULL || argument_pointer != NULL,
		"Function must get at least one pointer"
	);

	if ( function_pointer != NULL && cr_current_handle.function_pointer == NULL ) {
		int result = atexit( cr_cleanup_wrapper );
		if( assert_check_m( result == 0, "Exit handler registration failed" ) == true ) {
			cr_current_handle.function_pointer = function_pointer;
			cr_registered_atexit = true;
		}
	}
	if ( argument_pointer != NULL ) cr_current_handle.argument_pointer = argument_pointer;
}

/* Function:
 * internal callback that must call registered cleanup function
 */
static void cr_cleanup_wrapper(void) {
	if( assert_check_m(
			cr_current_handle.function_pointer != NULL && cr_registered_atexit == true,
			"Invalid exit callback state"
		) == true)
	{
		(*cr_current_handle.function_pointer)( cr_current_handle.argument_pointer );
	}
}
