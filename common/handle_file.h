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

#ifndef HANDLE_FILE_H
#define HANDLE_FILE_H

#	include <stdint.h>	/* uint_least64_t	*/
#	include <stddef.h>	/* size_t			*/
#	include <stdbool.h>	/* bool				*/

/* Function:
 * check if file exists at the given path
 *
 * Parameters:
 * path_pointer	- path to the object
 *
 * Returns:
 * true			- file exists
 * false		- failed to find a file
 */
bool hf_file_is_exists( const char * restrict path_pointer );
/* Function:
 * read a file into a heap-allocated buffer
 *
 * Parameters:
 * path_pointer		- path to the object
 * out_data_pointer	- save result data
 * length_pointer	- save the amount of bytes read
 *
 * Returns:
 * 0				- file allocated into the heap
 * -1				- failed to close a file
 * > 0				- wrong parameters, read, save failed
 */
int hf_file_read( const char * restrict path_pointer, char * restrict * restrict out_data_pointer, size_t * restrict length_pointer );
/* Function:
 * copy a file from source to destination
 *
 * Parameters:
 * path_pointer_source		- path to the object
 * path_pointer_destination	- path to destination object
 *
 * Returns:
 * 0						- file copied successfully
 * -1						- failed to close a file
 * > 0						- wrong parameters, open source/destination, read, write, copy failed
 */
int hf_file_copy( const char * restrict path_pointer_source, const char * restrict path_pointer_destination );
/* Function:
 * write data to a file (overwrites existing content)
 *
 * Parameters:
 * path_pointer	- path to the object
 * data_pointer	- data to write in file
 * length		- amount of bytes to write
 *
 * Returns:
 * 0			- data written successfully
 * -1			- failed to close a file
 * > 0			- wrong parameters, open, write, path conversion/resolution
 */
int hf_file_write( const char * restrict path_pointer, const char * restrict data_pointer, size_t length );
/* Function:
 * append data to the end of a file
 *
 * Parameters:
 * path_pointer	- path to the object
 * data_pointer	- data to write in file
 * length		- amount of bytes to append
 *
 * Returns:
 * 0			- data appended successfully
 * -1			- file closure failed
 * > 0			- wrong parameters, open, write, path conversion/resolution
 */
int hf_file_append( const char * restrict path_pointer, const char * restrict data_pointer, size_t length );
/* Function:
 * delete a file
 *
 * Parameters:
 * path_pointer	- path to the object
 *
 * Returns:
 * 0			- file deleted successfully
 * -1			- file closure failed
 * > 0			- wrong parameters, path conversion/resolution, deletion failed
 */
int hf_file_delete( const char * restrict path_pointer );
/* Function:
 * get the size of a file
 *
 * Parameters:
 * path_pointer		- path to the object
 * out_size_pointer	- save result data
 *
 * Returns:
 * 0				- file size retrieved
 * -1				- file closure failed
 * > 0				- wrong parameters, path conversion/resolution, extra large size, not a file
 */
int hf_file_size_get( const char * restrict path_pointer, uint_least64_t * restrict out_size_pointer );
/* Function:
 * rename or move a file
 *
 * Parameters:
 * path_pointer_old	- path to an old object
 * path_pointer_new	- path to a new object
 *
 * Returns:
 * 0				- file moved, renamed successfully
 * -1				- file closure failed
 * > 0				- wrong parameters, path conversion/resolution, move/rename,
 *						cross-device copy/delete failed
 */
int hf_file_rename_move( const char * restrict path_pointer_old, const char * restrict path_pointer_new );
/* Function:
 * read a chunk of data from a file at a specific position into a provided buffer
 *
 * Parameters:
 * path_pointer				- path to the object
 * data_pointer				- buffer to store read data
 * to_read_amount			- amount of bytes to read
 * position_to_read_from	- bytes offset to read from
 * out_read_amount_pointer	- storage of actual read bytes
 *
 * Returns:
 * 0						- chunk read successfully
 * -1						- file closure failed
 * > 0						- wrong parameters, open, seek, read, path conversion/resolution failed
 */
int hf_file_chunk_read( const char * restrict path_pointer, char * restrict data_pointer, size_t to_read_amount, uint_least64_t position_to_read_from, size_t * restrict out_read_amount_pointer );
/* Function:
 * write a chunk of data to a file at a specific position
 *
 * Parameters:
 * path_pointer					- path to the object
 * data_pointer					- data to write in file
 * to_write_amount				- amount of bytes to write
 * position_to_write_from		- bytes offset to write from
 * out_written_amount_pointer	- storage of actual written bytes
 *
 * Returns:
 * 0							- chunk written successfully
 * -1							- file closure failed
 * > 0							- wrong parameters, seek, write, path conversion/resolution,
 *									open failed
 */
int hf_file_chunk_write( const char * restrict path_pointer, const char * restrict data_pointer, size_t to_write_amount, uint_least64_t position_to_write_from, size_t * out_written_amount_pointer );

#endif /* HANDLE_FILE_H */
