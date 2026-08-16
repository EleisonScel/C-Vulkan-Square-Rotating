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

#ifndef WOEM_WRITE_OUT_ERROR_MESSAGE_H
#define	WOEM_WRITE_OUT_ERROR_MESSAGE_H

#  include <stdbool.h>	/* bool	*/

/* Function:
 * free all collected error messages and reset storage which should NOT happen
 */
void woem_clear(void);
/* Function:
 * shrink dynamic array of errors to its real size (amount if that's allocated)
 */
void woem_shrink(void);

/* Function:
 * format the error message out
 *
 * Parameters:
 * format_pointer, ...	- format string and its arguments
 *
 * Returns:
 * true					- saved error message
 * false				- saving error message failed, out of memory
 */
bool woem_push( const char * restrict format_pointer, ... );
/* Function:
 * push an allocated error message into the error messages array
 *
 * Parameters:
 * error_message_pointer	- pointer to an error message
 *
 * Returns:
 * true						- error message written successfully
 * false					- incorrect argument, not enough space to save error message;
 *								error message freed
 */
bool woem_push_raw( char * restrict error_message_pointer );

/* Function:
 * return last error message and remove it from the list
 *
 * Parameters:
 * out_must_be_freed_pointer	- pointer telling if the gotten string shall be freed or not
 *
 * Returns:
 * pointer						- the error string which was found
 * NULL							- no errors remain or no out_must_be_freed_pointer found
 */
char * woem_pop( bool * restrict out_must_be_freed_pointer );

#endif /* WOEM_WRITE_OUT_ERROR_MESSAGE_H */
