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

/* Portable overflow-checked integer addition.
 *
 * Every sa_ovf_add_<type> function adds two values of given type without
 * undefined behaviour, unlike the built-in '+' operator.
 *
 * Precondition:
 * result_pointer	- isn't NULL
 *
 * Arguments:
 * first			- first summand
 * second			- second summand
 * result_pointer	- pointer to store the sum
 *
 * Returns:
 * true				- overflow occurred, result pointer is NULL
 * false			- addition completed successfully
 */

#pragma once

#ifndef SAFE_ADDITION_H
#define SAFE_ADDITION_H

#	include "assert_m.h"/* assert_check_m */

#	include <stdint.h>	/* uint8_t	*/
#	include <stddef.h>	/* size_t	*/
#	include <stdbool.h>	/* bool		*/

#	ifndef __has_builtin
#		define __has_builtin(x) 0
#	endif

#	if defined(__STDC_VERSION__) && (__STDC_VERSION__ + 0L) >= 202311L &&\
		defined(__has_include) && __has_include(<stdckdint.h>)
#		define SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD 1
#		include <stdckdint.h>
#	else
#		define SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD 0
#	endif /* stdckdint.h */

#	if (__has_builtin(__builtin_add_overflow) != 0)
#		define SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN 1
#	else
#		define SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN 0
#	endif


#	if defined(UINT8_MAX)
static inline bool sa_ovf_add_uint8_t(
		uint8_t first, uint8_t second, uint8_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#		else

	uint_fast16_t temporary = (uint_fast16_t) first + (uint_fast16_t) second;
	if ( temporary > UINT8_MAX )
		return true;

	*result_pointer = (uint8_t) temporary;
	return false;

#		endif
}
#	endif /* UINT8_MAX */

#	if defined(UINT8_MAX) && defined(INT8_MAX)
static inline bool sa_ovf_add_int8_t(
		int8_t first, int8_t second, int8_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#		else

	int_fast16_t temporary = (int_fast16_t) first + (int_fast16_t) second;
	if ( temporary > INT8_MAX || temporary < INT8_MIN )
		return true;

	*result_pointer = (int8_t) temporary;
	return false;

#		endif
}
#	endif /* INT8_MAX */

#	if defined(UINT16_MAX)
static inline bool sa_ovf_add_uint16_t(
		uint16_t first, uint16_t second, uint16_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#		else

	uint_fast32_t temporary = (uint_fast32_t) first + (uint_fast32_t) second;
	if ( temporary > UINT16_MAX )
		return true;

	*result_pointer = (uint16_t) temporary;
	return false;

#		endif
}
#	endif /* UINT16_MAX */

#	if defined(UINT16_MAX) && defined(INT16_MAX)
static inline bool sa_ovf_add_int16_t(
		int16_t first, int16_t second, int16_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#		else

	int_fast32_t temporary = (int_fast32_t) first + (int_fast32_t) second;
	if ( temporary > INT16_MAX || temporary < INT16_MIN)
		return true;

	*result_pointer = (int16_t) temporary;
	return false;

#		endif
}
#	endif /* UINT16_MAX */

#	if defined(UINT32_MAX)
static inline bool sa_ovf_add_uint32_t(
		uint32_t first, uint32_t second, uint32_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#		else

	uint_fast64_t temporary = (uint_fast64_t) first + (uint_fast64_t) second;
	if ( temporary > UINT32_MAX )
		return true;

	*result_pointer = (uint32_t) temporary;
	return false;

#		endif
}
#	endif /* UINT32_MAX */

#	if defined(UINT32_MAX) && defined(INT32_MAX)
static inline bool sa_ovf_add_int32_t(
		int32_t first, int32_t second, int32_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#		else

	int_fast64_t temporary = (int_fast64_t) first + (int_fast64_t) second;
	if ( temporary > INT32_MAX || temporary < INT32_MIN)
		return true;

	*result_pointer = (int32_t) temporary;
	return false;

#		endif
}
#	endif /* UINT32_MAX */

#	if defined(UINT64_MAX)
static inline bool sa_ovf_add_uint64_t(
		uint64_t first, uint64_t second, uint64_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#		else

	if (first > UINT64_MAX - second)
		return true;

	*result_pointer = first + second;
	return false;

#		endif
}
#	endif /* UINT64_MAX */

#	if defined(UINT64_MAX) && defined(INT64_MAX)
static inline bool sa_ovf_add_int64_t(
		int64_t first, int64_t second, int64_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#		else

	if ((first > 0 && second > INT64_MAX - first) ||
		(first < 0 && second < INT64_MIN - first) )
		return true;

	*result_pointer = first + second;
	return false;

#		endif
}
#	endif /* UINT64_MAX */

static inline bool sa_ovf_add_size_t(
		size_t first, size_t second, size_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#	elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#	else

	if (first > SIZE_MAX - second)
		return true;

	*result_pointer = first + second;
	return false;

#	endif
}

static inline bool sa_ovf_add_ptrdiff_t(
		ptrdiff_t first, ptrdiff_t second, ptrdiff_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#	elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#	else

	if ((first > 0 && second > PTRDIFF_MAX - first) ||
		(first < 0 && second < PTRDIFF_MIN - first) )
		return true;

	*result_pointer = first + second;
	return false;

#	endif
}

static inline bool sa_ovf_add_uintmax_t(
		uintmax_t first, uintmax_t second, uintmax_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#	elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#	else

	if (first > UINTMAX_MAX - second)
		return true;

	*result_pointer = first + second;
	return false;

#	endif
}

static inline bool sa_ovf_add_intmax_t(
		intmax_t first, intmax_t second, intmax_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#	elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#	else

	if ((first > 0 && second > INTMAX_MAX - first) ||
		(first < 0 && second < INTMAX_MIN - first) )
		return true;

	*result_pointer = first + second;
	return false;

#	endif
}

#	ifdef UINTPTR_MAX
static inline bool sa_ovf_add_uintptr_t(
		uintptr_t first, uintptr_t second, uintptr_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#		else

	if (first > UINTPTR_MAX - second)
		return true;

	*result_pointer = first + second;
	return false;

#		endif
}
#	endif /* UINTPTR_MAX */

#	ifdef INTPTR_MAX
static inline bool sa_ovf_add_intptr_t(
		intptr_t first, intptr_t second, intptr_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_ADDITION_STANDARD == 1

	return ckd_add( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_ADDITION_BUILTIN == 1

	return __builtin_add_overflow( first, second, result_pointer );

#		else

	if ((first > 0 && second > INTPTR_MAX - first) ||
		(first < 0 && second < INTPTR_MIN - first) )
		return true;

	*result_pointer = first + second;
	return false;

#		endif
}
#	endif /* INTPTR_MAX */

#endif /* SAFE_ADDITION_H */
