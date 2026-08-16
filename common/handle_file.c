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

#include "handle_file.h"

#include "assert_m.h"				/* assert_check_m	*/
#include "safe_multiplication.h"	/* sa_ovf_mul_size_t*/
#include "write_out_error_message.h"/* woem_push		*/

#include <stdio.h>					/* fopen	*/
#include <stddef.h>					/* size_t	*/
#include <stdlib.h>					/* malloc	*/
#include <stdint.h>					/* SIZE_MAX	*/
#include <stdbool.h>				/* bool		*/

#include <errno.h>					/* errno	*/
#include <string.h>					/* strerror	*/
#include <inttypes.h>				/* PRIu64	*/

enum {
	RF_ERROR_CLOSE				= -1,
	RF_SUCCESS					= 0,

	RF_ERROR_NULL_PARAMETERS,
	RF_ERROR_MEMORY_ALLOCATION,

	RF_ERROR_OPEN,
	RF_ERROR_READ,
	RF_ERROR_WRITE,
	RF_ERROR_SEEK,
	RF_ERROR_TELL,

	RF_ERROR_STAT,
	RF_ERROR_ATTRIBUTES,
	RF_ERROR_NOT_A_FILE,
	RF_ERROR_LARGE_SIZE,
	RF_ERROR_ACCESS_DENIED,
	RF_ERROR_UTF_CONVERSION,
	RF_ERROR_PATH_IS_TOO_LONG,

	RF_ERROR_COPY,
	RF_ERROR_DELETE,
	RF_ERROR_RENAME_MOVE,
};

#if defined(RF_FORCE_WINDOWS) && !defined(_WIN32)
#	error "Windows API isn't available, remove RF_FORCE_WINDOWS"
#elif defined(RF_FORCE_WINDOWS)
#	define RF_BACKEND_WINDOWS
#elif defined(RF_FORCE_STAT)
#	define RF_BACKEND_STAT
#elif defined(RF_FORCE_FSEEKO)
#	define RF_BACKEND_FSEEKO
#elif defined(RF_FORCE_STANDARD)
#	define RF_BACKEND_STANDARD
#elif defined(_WIN32)
#	define RF_BACKEND_WINDOWS
#elif defined(__unix__) || defined(__unix) || defined(__APPLE__)
#	define RF_BACKEND_STAT
#elif defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE + 0L) >= 200112L
#	define RF_BACKEND_FSEEKO
#else
#	define RF_BACKEND_STANDARD
#endif /* RF_BACKEND_<type> */

#if defined(RF_BACKEND_WINDOWS)
#	define WIN32_LEAN_AND_MEAN
#	define NOMINMAX
#	include <wchar.h>	/* wchar_t				*/
#	include <windows.h>	/* GetFileAttributesExW	*/
#	include <io.h>		/* _wfopen				*/
#	include <limits.h>	/* CHAR_BIT				*/
static_assert_m( sizeof(wchar_t) > 1, "wchar_t must be larger than 1 byte" );
static_assert_m( sizeof(DWORD) * CHAR_BIT >= 32, "DWORD must be able to hold 32 bits" );
static_assert_m(
	sizeof(DWORD) <= sizeof(uint_least64_t),
	"uint_least64_t must be able to hold any DWORD value"
);
#	define RF_WIDE_PATH_BUFFER_MAXIMAL_SIZE		32767
#	define RF_UNIVERSAL_PREFIX					L"\\\\\\\\?\\"
#	define RF_UNIVERSAL_PREFIX_LENGTH			(sizeof("\\\\\\\\?\\") - 1)
#	define RF_WIDE_PATH_PREFIX_LENGTH			(sizeof("\\\\?\\") - 1)
#	define RF_UNC_REPLACE_PREFIX				L"?\\UNC"
#	define RF_UNC_POSITION_TO_REPLACE_FROM		(sizeof("?\\UNC") - 2)
#	define RF_UNC_REPLACE_PREFIX_LENGTH			(sizeof("?\\UNC") - 1)
#	define RF_UNIVERSAL_WIDE_PATH_BUFFER_SIZE\
		(RF_WIDE_PATH_BUFFER_MAXIMAL_SIZE + RF_UNIVERSAL_PREFIX_LENGTH)
#else
#	define RF_TEMPORARY_COPY_BUFFER_SIZE (64 * 1024) /* 64kb */
#	if defined(RF_BACKEND_STANDARD)
#		include <limits.h>	/* LONG_MAX	*/
#	endif
#	ifndef _FILE_OFFSET_BITS
#		define _FILE_OFFSET_BITS 64
#	elif _FILE_OFFSET_BITS < 64
#		ifndef RF_NO_WARNING_FILE_OFFSET_BITS
#			if defined(_MSC_VER)
#				pragma message("_FILE_OFFSET_BITS defined"\
					"and less than 64. Files larger than 2GB may fail to open.")
#			else
#				warning "_FILE_OFFSET_BITS defined"\
						"and less than 64. Files larger than 2GB may fail to open."
#			endif
#		endif /* RF_NO_WARNING_FILE_OFFSET_BITS */
#	endif
#	if defined(RF_BACKEND_STAT) || defined(RF_BACKEND_FSEEKO)
#		include <sys/types.h>	/* off_t*/
#	endif
#	if defined(RF_BACKEND_STAT)
#		include <sys/stat.h>	/* stat	*/
#	endif
#endif /* RF_BACKEND_WINDOWS */

static bool hf_path_has_wide_prefix( const wchar_t path_array[static 4] );
static bool hf_path_has_universal_naming_convention_prefix( const wchar_t path_array[static 2] );

static int hf_file_seek( const char * restrict function_name_pointer, FILE * restrict file_pointer, const char * restrict path_pointer, uint_least64_t position_to_read_from );
static int hf_file_open( const char * restrict file_path_pointer, const char * restrict mode_pointer, FILE * restrict * restrict out_file_pointer );
static int hf_file_open_for_read( const char * restrict path_pointer, FILE * restrict * restrict out_file_pointer, size_t * restrict out_file_size_pointer );
static int hf_file_copy_internal( const char * restrict function_name_pointer, const char * restrict path_pointer_source, const char * restrict path_pointer_destination );
static int hf_file_close_internal( const char * restrict function_name_pointer, FILE * file_pointer, const char * restrict file_path_pointer, int error_code_current );
static int hf_file_write_internal( const char * restrict file_path_pointer, const char * restrict data_pointer, size_t length, const char * restrict mode_pointer, const char * restrict function_name_pointer );
static int hf_file_get_information( const char * restrict function_name_pointer, const char * restrict path_pointer, uint_least64_t * restrict out_file_size_pointer, FILE * restrict * restrict out_file_pointer );
#ifdef RF_BACKEND_WINDOWS
static int hf_resolve_path_windows( const char * restrict path_pointer, wchar_t * restrict wide_path_pointer, wchar_t * restrict * restrict out_wide_path_pointer );
#elif defined(RF_BACKEND_FSEEKO) || defined(RF_BACKEND_STANDARD)
static int hf_file_seek_get_size( const char * restrict function_name_pointer, FILE * restrict file_pointer, const char * restrict file_path_pointer, uint_least64_t * restrict out_file_size_pointer, bool needed_rewind );
#endif /* RF_BACKEND_WINDOWS */

int hf_file_read(
		const char * restrict path_pointer, char * restrict * restrict out_data_pointer,
		size_t * restrict length_pointer
	)
{
	if( assert_check_m( path_pointer	!= NULL,"Filepath failed to find"		) == false ||
		assert_check_m( out_data_pointer!= NULL,"No place to write output found") == false)
		return RF_ERROR_NULL_PARAMETERS;

	*out_data_pointer = NULL;
	if ( length_pointer != NULL ) *length_pointer = 0;

	FILE	* file_pointer	= NULL;
	char	* text_pointer	= NULL;
	size_t	file_size		= 0,
			total_read_bytes= 0;

	int error_code = hf_file_open_for_read( path_pointer, &file_pointer, &file_size );
	if ( error_code != RF_SUCCESS ) return error_code;

	text_pointer = malloc( file_size + 1 );
	if ( text_pointer == NULL ) {
		woem_push( "(hf_file_read) lack of memory to allocate %zu bytes", file_size );
		error_code = RF_ERROR_MEMORY_ALLOCATION;
		goto cleanup;
	}

	total_read_bytes = fread( text_pointer, 1, file_size, file_pointer );

	if ( total_read_bytes != file_size ) {
		char * temporary_pointer = NULL;
		if ( ferror( file_pointer ) != 0 ) {
			woem_push(
				"(hf_file_read) I/O error while reading (%s) file (%s)",
				path_pointer, strerror(errno)
			);
			error_code = RF_ERROR_READ;
			goto cleanup;
		}
		if ( feof( file_pointer ) == 0 ) {
			woem_push(
				"(hf_file_read) reading stopped prematurely without EOF or I/O error file (%s)",
				path_pointer
			);
			error_code = RF_ERROR_READ;
			goto cleanup;
		}
		if ( total_read_bytes == 0 && file_size > 0 ) {
			woem_push(
				"(hf_file_read) now empty file (%s) expected with %zu bytes",
				path_pointer, file_size
			);
			error_code = RF_ERROR_READ;
			goto cleanup;
		}
		temporary_pointer = realloc( text_pointer, total_read_bytes + 1 );
		if ( temporary_pointer != NULL )
			text_pointer = temporary_pointer;
	}

	text_pointer[total_read_bytes] = '\0';

	if ( length_pointer != NULL )
		*length_pointer = total_read_bytes;
	*out_data_pointer = text_pointer;

cleanup:
	if ( error_code != RF_SUCCESS ) {
		if ( text_pointer != NULL )
			free( text_pointer );
		if ( out_data_pointer != NULL )
			*out_data_pointer = NULL;
		if ( length_pointer )
			*length_pointer = 0;
	}
	error_code = hf_file_close_internal(
		"hf_file_read", file_pointer, path_pointer, error_code
	);

	return error_code;
}

int hf_file_write(
		const char * restrict path_pointer, const char * restrict data_pointer, size_t length
	)
{
	if( assert_check_m( path_pointer != NULL, "Filepath failed to find"		)== false ||
		assert_check_m( data_pointer != NULL, "Data to write failed to find")== false)
		return RF_ERROR_NULL_PARAMETERS;

	return hf_file_write_internal( path_pointer, data_pointer, length, "wb", "hf_file_write" );
}

int hf_file_append(
		const char * restrict path_pointer, const char * restrict data_pointer, size_t length
	)
{
	if( assert_check_m( path_pointer != NULL, "Filepath failed to find"		) == false ||
		assert_check_m( data_pointer != NULL, "Data to write failed to find") == false)
		return RF_ERROR_NULL_PARAMETERS;

	return hf_file_write_internal( path_pointer, data_pointer, length, "ab", "hf_file_append" );
}

int hf_file_delete( const char * restrict path_pointer ) {
	if ( assert_check_m( path_pointer != NULL, "No path found" ) == false )
		return RF_ERROR_NULL_PARAMETERS;

#ifdef RF_BACKEND_WINDOWS

	wchar_t wide_path_buffer[RF_UNIVERSAL_WIDE_PATH_BUFFER_SIZE] = RF_UNIVERSAL_PREFIX;
	wchar_t * wide_path_buffer_pointer = wide_path_buffer + RF_UNIVERSAL_PREFIX_LENGTH;

	int error_code = hf_resolve_path_windows(
		path_pointer, wide_path_buffer, &wide_path_buffer_pointer
	);
	if ( error_code != RF_SUCCESS ) return error_code;

	if ( DeleteFileW( wide_path_buffer_pointer ) == 0 ) {
		woem_push(
			"(hf_file_delete) failed to delete file (%s) (%lu)", path_pointer, GetLastError()
		);
		return RF_ERROR_DELETE;
	}

#else

	if ( remove( path_pointer ) != 0 ) {
		woem_push(
			"(hf_file_delete) failed to delete file (%s) (%s)", path_pointer, strerror(errno)
		);
		return RF_ERROR_DELETE;
	}

#endif /* RF_BACKEND_WINDOWS */

	return RF_SUCCESS;
}

int hf_file_rename_move(
		const char * restrict path_pointer_old, const char * restrict path_pointer_new
	)
{
	if( assert_check_m( path_pointer_old != NULL, "No old path found" ) == false ||
		assert_check_m( path_pointer_new != NULL, "No new path found" ) == false )
		return RF_ERROR_NULL_PARAMETERS;

#ifdef RF_BACKEND_WINDOWS

	wchar_t wide_path_buffer_old[RF_UNIVERSAL_WIDE_PATH_BUFFER_SIZE] = RF_UNIVERSAL_PREFIX;
	wchar_t * wide_path_buffer_pointer_old = wide_path_buffer_old + RF_UNIVERSAL_PREFIX_LENGTH;

	int error_code = hf_resolve_path_windows(
		path_pointer_old, wide_path_buffer_old, &wide_path_buffer_pointer_old
	);
	if ( error_code != RF_SUCCESS ) return error_code;

	wchar_t wide_path_buffer_new[RF_UNIVERSAL_WIDE_PATH_BUFFER_SIZE] = RF_UNIVERSAL_PREFIX;
	wchar_t * wide_path_buffer_pointer_new = wide_path_buffer_new + RF_UNIVERSAL_PREFIX_LENGTH;

	error_code = hf_resolve_path_windows(
		path_pointer_new, wide_path_buffer_new, &wide_path_buffer_pointer_new
	);
	if ( error_code != RF_SUCCESS ) return error_code;

	if( MoveFileExW(
			wide_path_buffer_pointer_old, wide_path_buffer_pointer_new,
			MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED
		) == 0 )
	{
		woem_push(
			"(hf_file_rename_move) failed to move (%s) to (%s) (%lu)",
			path_pointer_old, path_pointer_new, GetLastError()
		);
		return RF_ERROR_RENAME_MOVE;
	}

#else

	if ( rename( path_pointer_old, path_pointer_new ) != 0 ) {

#	ifdef EXDEV

		if ( errno == EXDEV ) {
			int error_safed_operation = hf_file_copy_internal(
				"hf_file_rename_move", path_pointer_old, path_pointer_new
			);
			if ( error_safed_operation != RF_SUCCESS )
				return error_safed_operation;

			if ( remove( path_pointer_old ) != 0 ) {
				woem_push(
					"(hf_file_rename_move) file (%s) failed to delete after copying (%s)",
					path_pointer_old, strerror(errno)
				);
				return RF_ERROR_DELETE;
			}
			return RF_SUCCESS;
		}

#	endif /* EXDEV */

		woem_push(
			"(hf_file_rename_move) failed to move (%s) to (%s) (%s)",
			path_pointer_old, path_pointer_new, strerror(errno)
		);
		return RF_ERROR_RENAME_MOVE;
	}

#endif /* RF_BACKEND_WINDOWS */

	return RF_SUCCESS;
}

int hf_file_copy(
		const char * restrict path_pointer_source,
		const char * restrict path_pointer_destination
	)
{
	if( assert_check_m(path_pointer_source		!= NULL, "No source path found"	) == false ||
		assert_check_m(path_pointer_destination	!= NULL, "No new path found"	) == false )
		return RF_ERROR_NULL_PARAMETERS;
	return hf_file_copy_internal("hf_file_copy", path_pointer_source, path_pointer_destination);
}

bool hf_file_is_exists( const char * restrict path_pointer ) {
	if ( assert_check_m( path_pointer != NULL, "No path found" ) == false )
		return false;

#ifdef RF_BACKEND_WINDOWS

	wchar_t wide_path_buffer[RF_UNIVERSAL_WIDE_PATH_BUFFER_SIZE] = RF_UNIVERSAL_PREFIX;
	wchar_t * wide_path_buffer_pointer = wide_path_buffer + RF_UNIVERSAL_PREFIX_LENGTH;

	if ( hf_resolve_path_windows(
			path_pointer, wide_path_buffer, &wide_path_buffer_pointer
		) != RF_SUCCESS)
		return false;

	WIN32_FILE_ATTRIBUTE_DATA file_information;
	if( GetFileAttributesExW( wide_path_buffer_pointer, GetFileExInfoStandard, &file_information
		) == 0)
		return false;

	if ( (file_information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 )
		return false;

#elif defined(RF_BACKEND_STAT)

	struct stat file_status;
	if ( stat( path_pointer, &file_status ) != 0 )
		return false;

	if ( S_ISREG( file_status.st_mode ) == 0 )
		return false;

#else

	FILE * file_pointer = fopen( path_pointer, "rb" );
	if ( file_pointer != NULL )
		fclose( file_pointer );
	else return false;

#endif

	return true;
}

int hf_file_size_get(
		const char * restrict path_pointer, uint_least64_t * restrict out_file_size_pointer
	)
{
	if( assert_check_m(path_pointer			!= NULL, "No path found"			) == false ||
		assert_check_m(out_file_size_pointer!= NULL, "No place to write size found") == false )
		return RF_ERROR_NULL_PARAMETERS;

	return hf_file_get_information(
		"hf_file_size_get", path_pointer, out_file_size_pointer, NULL
	);
}

int hf_file_chunk_read(
		const char * restrict path_pointer, char * restrict data_pointer,
		size_t to_read_amount, uint_least64_t position_to_read_from,
		size_t * restrict out_read_amount_pointer
	)
{
	if( assert_check_m( path_pointer != NULL, "Filepath failed to find"		) == false ||
		assert_check_m( data_pointer != NULL, "Data to read failed to find"	) == false)
		return RF_ERROR_NULL_PARAMETERS;

	if ( out_read_amount_pointer != NULL )
		*out_read_amount_pointer = 0;

	FILE * file_pointer = NULL;
	size_t bytes_read = 0;

	int error_code = hf_file_open( path_pointer, "rb", &file_pointer );
	if( error_code != RF_SUCCESS )
		return error_code;

	if ( to_read_amount == 0)
		goto cleanup;

	error_code = hf_file_seek(
		"hf_file_chunk_read", file_pointer, path_pointer, position_to_read_from
	);
	if ( error_code != RF_SUCCESS )
		goto cleanup;

	bytes_read = fread( data_pointer, 1, to_read_amount, file_pointer );
	if ( ferror( file_pointer ) != 0 ) {
		woem_push(
			"(hf_file_chunk_read) I/O error while reading (%s) file (%s)",
			path_pointer, strerror(errno)
		);
		error_code = RF_ERROR_READ;
		goto cleanup;
	}

	if ( out_read_amount_pointer != NULL )
		*out_read_amount_pointer = bytes_read;

cleanup:
	return hf_file_close_internal(
		"hf_file_chunk_read", file_pointer, path_pointer, error_code
	);
}

int hf_file_chunk_write(
		const char * restrict path_pointer, const char * restrict data_pointer,
		size_t to_write_amount, uint_least64_t position_to_write_from,
		size_t * out_written_amount_pointer
	)
{
	if( assert_check_m( path_pointer != NULL, "Filepath failed to find"		) == false ||
		assert_check_m( data_pointer != NULL, "Data to write failed to find") == false)
		return RF_ERROR_NULL_PARAMETERS;

	if ( out_written_amount_pointer != NULL )
		*out_written_amount_pointer = 0;

	FILE * file_pointer = NULL;
	size_t bytes_written = 0;

	int error_code = hf_file_open( path_pointer, "r+b", &file_pointer );
	if( error_code != RF_SUCCESS )
		return error_code;

	if ( to_write_amount == 0)
		goto cleanup;

	error_code = hf_file_seek(
		"hf_file_chunk_write", file_pointer, path_pointer, position_to_write_from
	);
	if ( error_code != RF_SUCCESS )
		goto cleanup;

	bytes_written = fwrite( data_pointer, 1, to_write_amount, file_pointer );
	if ( bytes_written != to_write_amount ) {
		woem_push(
			"(hf_file_chunk_write) I/O error while writing (%s) file (%s)",
			path_pointer, strerror(errno)
		);
		error_code = RF_ERROR_WRITE;
		goto cleanup;
	}

	if ( out_written_amount_pointer != NULL )
		*out_written_amount_pointer = bytes_written;

cleanup:
	return hf_file_close_internal(
		"hf_file_chunk_write", file_pointer, path_pointer, error_code
	);
}

#if defined(RF_BACKEND_FSEEKO) || defined(RF_BACKEND_STANDARD)
/* Function:
 * get file size and optionally rewind
 *
 * Parameters:
 * function_name_pointer- name of calling function
 * file_pointer			- file handle
 * file_path_pointer	- file path
 * out_file_size_pointer- pointer to store file size
 * needed_rewind		- rewind to the start needed at the end
 *
 * Returns:
 * 0					- file size retrieved successfully
 * >0					- seek, extra large size, getting position failed
 */
static int hf_file_seek_get_size(
		const char * restrict function_name_pointer, FILE * restrict file_pointer,
		const char * restrict file_path_pointer,
		uint_least64_t * restrict out_file_size_pointer, bool needed_rewind
	)
{
	assert_m( function_name_pointer	!= NULL, "No function name found"			);
	assert_m( file_pointer			!= NULL, "No file found"					);
	assert_m( file_path_pointer		!= NULL, "No file path found"				);
	assert_m( out_file_size_pointer	!= NULL, "No place to write file size found");

	*out_file_size_pointer = 0;
#	if defined(RF_BACKEND_FSEEKO)

	off_t file_size_temporary;
	if ( fseeko( file_pointer, 0, SEEK_END ) != 0 ) {
		woem_push(
			"(%s) end of the file (%s) failed to find", function_name_pointer, file_path_pointer
		);
		return RF_ERROR_SEEK;
	}

	file_size_temporary = ftello( file_pointer );
	if ( file_size_temporary < 0 ) {
		woem_push( "(%s) reading file size failed", function_name_pointer );
		return RF_ERROR_TELL;
	}

	*out_file_size_pointer = (uint_least64_t) file_size_temporary;
	if ( (off_t) *out_file_size_pointer != file_size_temporary ) {
		woem_push( "(%s) file size exceed memory limits", function_name_pointer );
		return RF_ERROR_LARGE_SIZE;
	}

	if ( needed_rewind == true ) {
		if ( fseeko( file_pointer, 0, SEEK_SET ) != 0 ) {
			woem_push(
				"(%s) start of the file (%s) failed to find",
				function_name_pointer, file_path_pointer
			);
			return RF_ERROR_SEEK;
		}
	}

#	else

	long int file_size_temporary;
	if ( fseek( file_pointer, 0, SEEK_END ) != 0 ) {
		woem_push(
			"(%s) end of the file (%s) failed to find",
			function_name_pointer, file_path_pointer
		);
		return RF_ERROR_SEEK;
	}

	file_size_temporary = ftell( file_pointer );
	if ( file_size_temporary < 0 ) {
		woem_push( "(%s) reading file size failed", function_name_pointer );
		return RF_ERROR_TELL;
	}

	*out_file_size_pointer = (uint_least64_t) file_size_temporary;
	if ( (long int) *out_file_size_pointer != file_size_temporary ) {
		woem_push( "(%s) file size exceed memory limits", function_name_pointer );
		return RF_ERROR_LARGE_SIZE;
	}

	if ( needed_rewind == true ) {
		if ( fseek( file_pointer, 0, SEEK_SET ) != 0 ) {
			woem_push(
				"(%s) start of the file (%s) failed to find",
				function_name_pointer, file_path_pointer
			);
			return RF_ERROR_SEEK;
		}
	}

#	endif /* RF_BACKEND_FSEEKO */

	return RF_SUCCESS;
}
#endif /* RF_BACKEND_FSEEKO || RF_BACKEND_STANDARD */

/* Function:
 * close a file and handle error
 *
 * Parameters:
 * function_name_pointer- name of calling function
 * file_pointer			- file handle
 * file_path_pointer	- file to path for errors
 * error_code_current	- error code before closing
 *
 * Returns:
 * error_code			- file closed successfully
 * -1					- file closure failed
 */
static int hf_file_close_internal(
		const char * restrict function_name_pointer, FILE * restrict file_pointer,
		const char * restrict file_path_pointer, int error_code_current
	)
{
	assert_m( function_name_pointer	!= NULL, "No function name found"	);
	assert_m( file_path_pointer		!= NULL, "No file path found"		);

	if ( file_pointer != NULL ) {
		if ( fclose( file_pointer ) != 0 ) {
			woem_push(
				"(%s) closing file (%s) failed (%s)",
				function_name_pointer, file_path_pointer, strerror(errno)
			);
			if( error_code_current == RF_SUCCESS )
				return RF_ERROR_CLOSE;
		}
	}

	return error_code_current;
}

/* Function:
 * internal implementation of file copying
 *
 * Parameters:
 * function_name_pointer	- name of calling function
 * path_pointer_source		- path to source file
 * path_pointer_destination	- path to destination file
 *
 * Returns:
 * 0						- file copied successfully
 * -1						- file closure failed
 * > 0						- open source/destination, read, write, copy, delete failed
 */
static int hf_file_copy_internal(
		const char * restrict function_name_pointer,
		const char * restrict path_pointer_source,
		const char * restrict path_pointer_destination
	)
{
	assert_m(path_pointer_source		!= NULL, "No source path found"		);
	assert_m(function_name_pointer		!= NULL, "No function name found"	);
	assert_m(path_pointer_destination	!= NULL, "No new path found"		);

#ifdef RF_BACKEND_WINDOWS

	wchar_t wide_path_buffer_source[RF_UNIVERSAL_WIDE_PATH_BUFFER_SIZE] = RF_UNIVERSAL_PREFIX;
	wchar_t *wide_path_buffer_pointer_source =
		wide_path_buffer_source + RF_UNIVERSAL_PREFIX_LENGTH;

	int error_code = hf_resolve_path_windows(
		path_pointer_source, wide_path_buffer_source, &wide_path_buffer_pointer_source
	);
	if ( error_code != RF_SUCCESS ) return error_code;

	wchar_t wide_path_buffer_destination[RF_UNIVERSAL_WIDE_PATH_BUFFER_SIZE] =
		RF_UNIVERSAL_PREFIX;
	wchar_t * wide_path_buffer_pointer_destination =
		wide_path_buffer_destination + RF_UNIVERSAL_PREFIX_LENGTH;

	error_code = hf_resolve_path_windows(
		path_pointer_destination, wide_path_buffer_destination,
		&wide_path_buffer_pointer_destination
	);
	if ( error_code != RF_SUCCESS ) return error_code;

	if( CopyFileW(wide_path_buffer_pointer_source, wide_path_buffer_pointer_destination, FALSE)
		== 0 )
	{
		woem_push(
			"(%s) failed to copy (%s) file to (%s) (%lu)",
			function_name_pointer, path_pointer_source, path_pointer_destination, GetLastError()
		);
		return RF_ERROR_COPY;
	}

#else

	FILE * file_pointer_source = fopen( path_pointer_source, "rb" );
	if ( file_pointer_source == NULL ) {
		woem_push(
			"(%s) failed to open (%s) to copy (%s)",
			function_name_pointer, path_pointer_source, strerror(errno)
		);
		return RF_ERROR_OPEN;
	}

	FILE * file_pointer_destination = fopen( path_pointer_destination, "wb" );
	if ( file_pointer_destination == NULL ) {
		woem_push(
			"(%s) failed to open (%s) to copy (%s)",
			function_name_pointer, path_pointer_destination, strerror(errno)
		);
		if ( fclose( file_pointer_source ) != 0) {
			woem_push(
				"(%s) source file (%s) failed to close (%s)",
				function_name_pointer, path_pointer_source, strerror(errno)
			);
		}
		return RF_ERROR_OPEN;
	}

	char file_buffer[RF_TEMPORARY_COPY_BUFFER_SIZE];
	size_t bytes_read;

	int error_safed_operation = RF_SUCCESS;
	while (
		(bytes_read =
			fread(file_buffer, 1, sizeof(file_buffer), file_pointer_source)) > 0 )
	{
		if( fwrite(file_buffer, 1, bytes_read, file_pointer_destination)
			!= bytes_read )
		{
			error_safed_operation = RF_ERROR_WRITE;
			woem_push(
				"(%s) file (%s) failed to write to (%s) (%s)", function_name_pointer,
				path_pointer_source, path_pointer_destination, strerror(errno)
			);
			break;
		}
	}

	if ( ferror( file_pointer_source ) != 0 ) {
		error_safed_operation = RF_ERROR_READ;
		woem_push(
			"(%s) source file (%s) failed to read (%s)",
			function_name_pointer, path_pointer_source, strerror(errno)
		);
	}

	error_safed_operation = hf_file_close_internal(
		function_name_pointer, file_pointer_source, path_pointer_source,
		error_safed_operation
	);
	error_safed_operation = hf_file_close_internal(
		function_name_pointer, file_pointer_destination, path_pointer_destination,
		error_safed_operation
	);

	if ( error_safed_operation != RF_SUCCESS ) {
		if ( remove( path_pointer_destination ) != 0 ) {
			woem_push(
				"(%s) failed to delete file (%s) (%s)",
				function_name_pointer, path_pointer_destination, strerror(errno)
			);
		}
		return error_safed_operation;
	}

#endif /* RF_BACKEND_WINDOWS */

	return RF_SUCCESS;
}

/* Function:
 * seek to a specific position in a file
 *
 * Parameters:
 * function_name_pointer- name of calling function for errors
 * file_pointer			- file handle
 * path_pointer			- path to file for errors
 * position_to_read_from- byte offset to seek
 *
 * Returns:
 * 0					- seeked successfully
 * > 0					- extra large size, seek failed
 */
static int hf_file_seek(
		const char * restrict function_name_pointer, FILE * restrict file_pointer,
		const char * restrict path_pointer, uint_least64_t position_to_read_from
	)
{
	assert_m( function_name_pointer	!= NULL, "No function name found"	);
	assert_m( file_pointer			!= NULL, "No file found"			);
	assert_m( path_pointer			!= NULL, "No path found"			);

#if defined(RF_BACKEND_WINDOWS)

	__int64 target_position = (__int64) position_to_read_from;
	if ( (uint_least64_t) target_position != position_to_read_from || target_position < 0 ) {
		woem_push(
			"(%s) position exceeds seek limits (%s)", function_name_pointer, path_pointer
		);
		return RF_ERROR_LARGE_SIZE;
	}
	if ( _fseeki64( file_pointer, target_position, SEEK_SET ) != 0 ) {
		woem_push( "(%s) failed to seek in file (%s)", function_name_pointer, path_pointer );
		return RF_ERROR_SEEK;
	}

#elif defined(RF_BACKEND_FSEEKO)

	off_t target_position = (off_t) position_to_read_from;
	if ( (uint_least64_t) target_position != position_to_read_from || target_position < 0 ) {
		woem_push(
			"(%s) position exceeds seek limits (%s)", function_name_pointer, path_pointer
		);
		return RF_ERROR_LARGE_SIZE;
	}
	if ( fseeko( file_pointer, target_position, SEEK_SET ) != 0 ) {
		woem_push( "(%s) failed to seek in file (%s)", function_name_pointer, path_pointer );
		return RF_ERROR_SEEK;
	}

#else /* RF_BACKEND_STANDARD */

	long int target_position = (long int) position_to_read_from;
	if ( (uint_least64_t) target_position != position_to_read_from || target_position < 0 ) {
		woem_push(
			"(%s) position exceeds seek limits (%s)", function_name_pointer, path_pointer
		);
		return RF_ERROR_LARGE_SIZE;
	}
	if ( fseek( file_pointer, target_position, SEEK_SET ) != 0 ) {
		woem_push( "(%s) failed to seek in file (%s)", function_name_pointer, path_pointer );
		return RF_ERROR_SEEK;
	}

#endif /* RF_BACKEND_WINDOWS */

	return RF_SUCCESS;
}

/* Function:
 * internal implementation of file append/writing
 *
 * Parameters:
 * file_path_pointer	- path to file
 * data_pointer			- data to write
 * length				- amount of bytes to write
 * mode_pointer			- file open mode
 * function_name_pointer- name of calling function for errors
 *
 * Returns:
 * 0					- file written successfully
 * -1					- file closure failed
 * > 0					- open, write failed
 */
static int hf_file_write_internal(
		const char * restrict file_path_pointer, const char * restrict data_pointer,
		size_t length,
		const char * restrict mode_pointer, const char * restrict function_name_pointer
	)
{
	assert_m( function_name_pointer	!= NULL, "No function name found"		);
	assert_m( file_path_pointer		!= NULL, "Filepath failed to find"		);
	assert_m( data_pointer			!= NULL, "Data to write failed to find"	);
	assert_m( mode_pointer			!= NULL, "No open mode found"			);

	FILE * file_pointer = NULL;

	int error_code = hf_file_open( file_path_pointer, mode_pointer, &file_pointer );
	if( error_code != RF_SUCCESS )
		return error_code;

	if ( length != 0 ) {
		size_t bytes_written = fwrite( data_pointer, 1, length, file_pointer );
		if ( bytes_written != length ) {
			woem_push(
				"(%s) I/O error while writing file (%s) (%s)",
				function_name_pointer, file_path_pointer, strerror(errno)
			);
			error_code = RF_ERROR_WRITE;
		}
	}

	return hf_file_close_internal(
		function_name_pointer, file_pointer, file_path_pointer, error_code
	);
}

/* Function:
 * open a file with a specific mode
 *
 * Parameters:
 * path_pointer			- path to the file
 * mode_pointer			- file open mode
 * out_file_pointer		- pointer to store file handle
 *
 * Returns:
 * 0					- file opened successfully
 * > 0					- open, path conversion/resolution failed
 */
static int hf_file_open(
		const char * restrict file_path_pointer, const char * restrict mode_pointer,
		FILE * restrict * restrict out_file_pointer
	)
{
	assert_m( file_path_pointer	!= NULL, "No path found"				);
	assert_m( mode_pointer		!= NULL, "No open mode found"			);
	assert_m( out_file_pointer	!= NULL, "No place to save file found"	);

	*out_file_pointer = NULL;

#ifdef RF_BACKEND_WINDOWS
	
	wchar_t wide_path_buffer[RF_UNIVERSAL_WIDE_PATH_BUFFER_SIZE] = RF_UNIVERSAL_PREFIX;
	wchar_t * wide_path_buffer_pointer = wide_path_buffer + RF_UNIVERSAL_PREFIX_LENGTH;

	int error_code = hf_resolve_path_windows(
		file_path_pointer, wide_path_buffer, &wide_path_buffer_pointer
	);
	if ( error_code != RF_SUCCESS ) return error_code;

	wchar_t wide_mode_array[4] = { 0 };
	for ( int position = 0; mode_pointer[position] != 0 && position < 3; ++position )
		wide_mode_array[position] = (wchar_t)mode_pointer[position];

	*out_file_pointer = _wfopen( wide_path_buffer_pointer, wide_mode_array );

#else

	*out_file_pointer = fopen( file_path_pointer, mode_pointer );

#endif

	if ( *out_file_pointer == NULL ) {
		woem_push(
			"(hf_file_open) opening file (%s) failed (%s)",
			file_path_pointer, strerror(errno)
		);
		return RF_ERROR_OPEN;
	}

	return RF_SUCCESS;
}

/* Function:
 * open a file for reading and retrieve its size
 *
 * Parameters:
 * path_pointer			- path to the object
 * out_file_pointer		- pointer to store file handle
 * out_file_size_pointer- pointer to store file size
 *
 * Returns:
 * 0					- file opened and size retrieved
 * -1					- file closure failed
 * > 0					- extra large size, getting size, open failed
 */
static int hf_file_open_for_read(
		const char * restrict path_pointer, FILE * restrict * restrict out_file_pointer,
		size_t * restrict out_file_size_pointer
	)
{
	assert_m( path_pointer			!= NULL, "No path found"					);
	assert_m( out_file_size_pointer	!= NULL, "No place to write file size found");
	assert_m( out_file_pointer		!= NULL, "No place to save file found"		);

	*out_file_pointer		= NULL;
	*out_file_size_pointer	= 0;

	uint_least64_t file_size_temporary = 0;
	int error_code = hf_file_get_information(
		"hf_file_open_for_read", path_pointer, &file_size_temporary, out_file_pointer
	);
	if ( error_code != RF_SUCCESS ) return error_code;

#	if SIZE_MAX <= UINT_LEAST64_MAX

	if ( file_size_temporary > (uint_least64_t) (SIZE_MAX - 1) ) {
		woem_push("(hf_file_open_for_read) file (%s) size exceed memory limits", path_pointer);
		if ( fclose(*out_file_pointer) != 0 ) {
			woem_push(
				"(hf_file_open_for_read) closing file (%s) failed (%s)",
				path_pointer, strerror(errno)
			);
		}
		*out_file_pointer = NULL;
		return RF_ERROR_LARGE_SIZE;
	}

#	endif

	*out_file_size_pointer = (size_t) file_size_temporary;

	return RF_SUCCESS;
}

#ifdef RF_BACKEND_WINDOWS
/* Function:
 * resolve a UTF-8 path to a wide character path for Windows API
 *
 * Parameters:
 * path_pointer			- UTF-8 path
 * wide_path_pointer	- pointer to store wide path
 * out_wide_path_pointer- pointer to store resolved wide path
 *
 * Returns:
 * 0					- path resolved
 * > 0					- path conversion/resolution failed
 */
static int hf_resolve_path_windows(
		const char * restrict path_pointer, wchar_t * restrict wide_path_pointer,
		wchar_t * restrict * restrict out_wide_path_pointer
	)
{
	assert_m( path_pointer			!= NULL, "No path found"					);
	assert_m( wide_path_pointer		!= NULL, "No wide path found"				);
	assert_m( out_wide_path_pointer	!= NULL, "No place to write wide path found");

	wchar_t * wide_path_buffer_pointer = wide_path_pointer + RF_UNIVERSAL_PREFIX_LENGTH;

	int wide_path_length = MultiByteToWideChar(
		CP_UTF8, 0, path_pointer, -1, wide_path_buffer_pointer,
		RF_WIDE_PATH_BUFFER_MAXIMAL_SIZE
	);
	if ( wide_path_length == 0 ) {
		wide_path_length = MultiByteToWideChar(
			CP_ACP, 0, path_pointer, -1, wide_path_buffer_pointer,
			RF_WIDE_PATH_BUFFER_MAXIMAL_SIZE
		);
		if ( wide_path_length == 0 ) {
			woem_push(
				"(hf_resolve_path_windows) failed to convert path (error %lu)", GetLastError()
			);
			return RF_ERROR_UTF_CONVERSION;
		}
	}

	assert_mf(
		wide_path_length <= RF_WIDE_PATH_BUFFER_MAXIMAL_SIZE,
		"too long path (%d >= %d)", wide_path_length, RF_WIDE_PATH_BUFFER_MAXIMAL_SIZE
	);

	if( wide_path_length >= MAX_PATH &&
		hf_path_has_wide_prefix( wide_path_buffer_pointer ) == false )
	{
		wchar_t wide_path_buffer_absolute[RF_WIDE_PATH_BUFFER_MAXIMAL_SIZE + 1];
		if( _wfullpath(
				wide_path_buffer_absolute, wide_path_buffer_pointer,
				RF_WIDE_PATH_BUFFER_MAXIMAL_SIZE
			) == NULL )
		{
			woem_push("(hf_resolve_path_windows) resolving path failed (%s)", strerror(errno));
			return RF_ERROR_PATH_IS_TOO_LONG;
		}
		size_t wide_path_length_absolute = wcslen( wide_path_buffer_absolute );
		if ( wide_path_length_absolute >= RF_WIDE_PATH_BUFFER_MAXIMAL_SIZE ) {
			woem_push(
				"(hf_resolve_path_windows) absolute path exceeds length limits (%zu > %d)",
				wide_path_length_absolute, RF_WIDE_PATH_BUFFER_MAXIMAL_SIZE
			);
			return RF_ERROR_PATH_IS_TOO_LONG;
		}

		size_t bytes_to_copy;
		if ( sa_ovf_mul_size_t(
				wide_path_length_absolute + 1, sizeof(wchar_t), &bytes_to_copy
			) == true )
		{
			woem_push( "(hf_resolve_path_windows) file path is too long to handle" );
			return RF_ERROR_PATH_IS_TOO_LONG;
		}
		memcpy(
			wide_path_buffer_pointer,
			wide_path_buffer_absolute,
			bytes_to_copy
		);
		wide_path_length = (int) wide_path_length_absolute + 1;
		if( hf_path_has_universal_naming_convention_prefix(wide_path_buffer_pointer) == false) {
			wide_path_buffer_pointer -= RF_WIDE_PATH_PREFIX_LENGTH;
		} else {
			wmemcpy(
				wide_path_buffer_pointer - RF_UNC_POSITION_TO_REPLACE_FROM,
				RF_UNC_REPLACE_PREFIX,
				RF_UNC_REPLACE_PREFIX_LENGTH
			);
			wide_path_buffer_pointer -= RF_UNIVERSAL_PREFIX_LENGTH;
		}
	}
	*out_wide_path_pointer = wide_path_buffer_pointer;
	return RF_SUCCESS;
}
#endif /* RF_BACKEND_WINDOWS */

/* Function:
 * check if path has a universal naming convention prefix
 *
 * Precondition:
 * path is at least 2 characters wide, has NO wide prefix
 *
 * Parameters:
 * path_array	- path to check
 *
 * Returns:
 * true			- universal naming convention prefix found
 * false		- failed to find
 */
static bool hf_path_has_universal_naming_convention_prefix(const wchar_t path_array[static 2]) {
	return ( path_array[0] == L'\\' && path_array[1] == L'\\' );
}

/* Function:
 * find the wide prefix
 *
 * Parameters:
 * path_array	- path to check
 *
 * Returns:
 * true			- wide prefix found
 * false		- failed to find
 */
static bool hf_path_has_wide_prefix( const wchar_t path_array[static 4] ) {
	return (path_array[0] == L'\\' && path_array[1] == L'\\' && path_array[3] == L'\\' &&
			(path_array[2] == L'?' || path_array[2] == L'.')
	);
}

/* Function:
 * get file size and optionally open its handle
 *
 * Parameters:
 * function_name_pointer	- name of calling function for errors
 * path_pointer				- path to file
 * out_file_size_pointer	- pointer to store file size
 * out_file_pointer			- pointer to store file handle
 *
 * Returns:
 * 0						- information retrieved successfully
 * -1						- file closure failed
 * > 0						- not a file, extra large size, path conversion/resolution, open,
 *								seek, getting position failed
 */
static int hf_file_get_information(
		const char * restrict function_name_pointer, const char * restrict path_pointer,
		uint_least64_t * restrict out_file_size_pointer,
		FILE * restrict * restrict out_file_pointer
	)
{
	assert_m( path_pointer			!= NULL, "No path found"					);
	assert_m( out_file_size_pointer	!= NULL, "No place to write file size found");

	*out_file_size_pointer = 0;
	if ( out_file_pointer != NULL )
		*out_file_pointer = NULL;

#ifdef RF_BACKEND_WINDOWS

	wchar_t wide_path_buffer[RF_UNIVERSAL_WIDE_PATH_BUFFER_SIZE] = RF_UNIVERSAL_PREFIX;
	wchar_t * wide_path_buffer_pointer = wide_path_buffer + RF_UNIVERSAL_PREFIX_LENGTH;

	int error_code = hf_resolve_path_windows(
		path_pointer, wide_path_buffer, &wide_path_buffer_pointer
	);
	if ( error_code != RF_SUCCESS ) return error_code;

	WIN32_FILE_ATTRIBUTE_DATA file_information;
	if( GetFileAttributesExW( wide_path_buffer_pointer, GetFileExInfoStandard, &file_information
		) == 0)
	{
		woem_push(
			"(%s) file (%s) attributes failed to get (%lu)",
			function_name_pointer, path_pointer, GetLastError()
		);
		return RF_ERROR_ATTRIBUTES;
	}

	if ( (file_information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ) {
		woem_push( "(%s) (%s) is a directory", function_name_pointer, path_pointer );
		return RF_ERROR_NOT_A_FILE;
	}

	*out_file_size_pointer = ((uint_least64_t) file_information.nFileSizeHigh << 32 ) |
		(uint_least64_t) file_information.nFileSizeLow;

	if ( out_file_pointer != NULL ) {
		*out_file_pointer = _wfopen( wide_path_buffer_pointer, L"rb" );
		if ( *out_file_pointer == NULL ) {
			woem_push(
				"(%s) file (%s) failed to open (%s)",
				function_name_pointer, path_pointer, strerror(errno)
			);
			return RF_ERROR_OPEN;
		}
	}

#elif defined(RF_BACKEND_STAT)

	struct stat file_status;
	if ( stat( path_pointer, &file_status ) != 0 ) {
		woem_push(
			"(%s) getting file (%s) status failed (%s)",
			function_name_pointer, path_pointer, strerror(errno)
		);
		return RF_ERROR_STAT;
	}

	if ( S_ISREG( file_status.st_mode ) == 0 ) {
		woem_push( "(%s) (%s) isn't a regular file", function_name_pointer, path_pointer );
		return RF_ERROR_NOT_A_FILE;
	}

	if ( file_status.st_size < 0 ) {
		woem_push( "(%s) negative file (%s) size", function_name_pointer, path_pointer );
		return RF_ERROR_STAT;
	}

	*out_file_size_pointer = (uint_least64_t) file_status.st_size;
	if ( (off_t) *out_file_size_pointer != file_status.st_size ) {
		woem_push( "(%s) file size exceed memory limits", function_name_pointer );
		return RF_ERROR_LARGE_SIZE;
	}

	if ( out_file_pointer != NULL ) {
		*out_file_pointer = fopen( path_pointer, "rb" );
		if ( *out_file_pointer == NULL ) {
			woem_push(
				"(%s) file (%s) failed to open (%s)",
				function_name_pointer, path_pointer, strerror(errno)
			);
			return RF_ERROR_OPEN;
		}
	}

#else

	FILE * file_pointer = fopen( path_pointer, "rb" );
	if ( file_pointer == NULL ) {
		woem_push(
			"(%s) file (%s) failed to open (%s)",
			function_name_pointer, path_pointer, strerror(errno)
		);
		return RF_ERROR_OPEN;
	}

	int error_code = hf_file_seek_get_size(
		function_name_pointer, file_pointer, path_pointer, out_file_size_pointer,
		(out_file_pointer != NULL)
	);
	if( error_code != RF_SUCCESS ) {
		if ( fclose( file_pointer ) != 0 ) {
			woem_push(
				"(%s) closing file (%s) failed (%s)",
				function_name_pointer, path_pointer, strerror(errno)
			);
		}
		return error_code;
	}

	if ( out_file_pointer != NULL ) {
		*out_file_pointer = file_pointer;
	} else if ( fclose( file_pointer ) != 0 ) {
		woem_push(
			"(%s) closing file (%s) failed (%s)",
			function_name_pointer, path_pointer, strerror(errno)
		);
		return RF_ERROR_CLOSE;
	}

#endif /* RF_BACKEND_WINDOWS */

	return RF_SUCCESS;
}
