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

#include "exit_print.h"

#include "assert_m.h"	/* assert_m */

#include <stdio.h>		/* fprintf		*/
#include <stdarg.h>		/* va_list		*/
#include <stdlib.h>		/* EXIT_FAILURE	*/
#include <stdbool.h>	/* bool			*/

static void ep_inner_print( const char * restrict format_pointer, va_list arguments );

void ep_exit_print( const char * restrict format_pointer, ... ) {
	assert_m( format_pointer != NULL, "No format string found" );

	va_list arguments;
	va_start( arguments, format_pointer );

	ep_inner_print( format_pointer, arguments );

	va_end( arguments );

	exit( EXIT_FAILURE );
}

void ep_exit_print_free( bool free_flag, char * restrict format_pointer, ... ) {
	assert_m( format_pointer != NULL, "No format string found" );

	va_list arguments;
	va_start( arguments, format_pointer );

	ep_inner_print( format_pointer, arguments );

	va_end( arguments );

	if ( format_pointer != NULL && free_flag == true ) free( format_pointer );

	exit( EXIT_FAILURE );
}

/* Function:
 * print an error message
 *
 * Parameters:
 * format_pointer	- format string
 * arguments		- format arguments
 */
static void ep_inner_print( const char * restrict format_pointer, va_list arguments ) {
	if ( format_pointer == NULL ) {
		(void) fprintf( stderr, "ERR: (ep_exit_print) no text to print\n" );
		return;
	}
	(void) fputs( "ERR: ", stderr );
	(void) vfprintf( stderr, format_pointer, arguments );
	(void) fputc( '\n', stderr );
}
