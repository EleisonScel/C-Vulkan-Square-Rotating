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

#ifndef CLEANUP_REGISTER_H
#define CLEANUP_REGISTER_H

/* Function:
 * register a cleanup function, its argument or both as an exit callback
 *
 * Parameters:
 * function_pointer	- function to call at exit
 * argument_pointer	- argument pass to the function
 */
void cr_register_cleanup_wrapper( void (*function_pointer)(void *), void * restrict argument_pointer );

#endif /* CLEANUP_REGISTER_H */
