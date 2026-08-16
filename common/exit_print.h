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

#ifndef EXIT_PRINT_H
#define EXIT_PRINT_H

#  include <stdbool.h> /* bool */

/* Function:
 * print an error message and exit with failure code
 *
 * Parameters:
 * format_pointer, ...	- string and its format arguments
 */
void ep_exit_print( const char * restrict format_pointer, ... );
/* Function:
 * print an error message and exit with failure code
 *
 * Parameters:
 * free_flag			- flag if the format string shall be freed
 * format_pointer, ...	- string and its format arguments
 */
void ep_exit_print_free( bool free_flag, char * restrict format_pointer, ... );

#endif/* EXIT_PRINT_H */
