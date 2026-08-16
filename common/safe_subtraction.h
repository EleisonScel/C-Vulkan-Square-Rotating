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

/* Portable overflow-checked integer subtraction.
 *
 * Every sa_ovf_sub_<type> function subtracts two values of the given type
 * without undefined behaviour, unlike the built-in '-' operator.
 *
 * Precondition:
 * result_pointer	- isn't NULL
 *
 * Arguments:
 * minuend			- value to subtract from
 * subtrahend		- value to subtract
 * result_pointer	- pointer to store the difference
 *
 * Returns:
 * true				- overflow occurred, result pointer is NULL
 * false			- subtraction completed successfully
 */

#pragma once

#ifndef SAFE_SUBTRACTION_H
#define SAFE_SUBTRACTION_H

#	include "assert_m.h"/* assert_check_m */

#	include <stdint.h>	/* uint8_t	*/
#	include <stddef.h>	/* size_t	*/
#	include <stdbool.h>	/* bool		*/

#	ifndef __has_builtin
#		define __has_builtin(x) 0
#	endif

#	if defined(__STDC_VERSION__) && (__STDC_VERSION__ + 0L) >= 202311L &&\
		defined(__has_include) && __has_include(<stdckdint.h>)
#		define SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD 1
#		include <stdckdint.h>
#	else
#		define SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD 0
#	endif /* stdckdint.h */

#	if (__has_builtin(__builtin_sub_overflow) != 0)
#		define SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN 1
#	else
#		define SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN 0
#	endif

#	if defined(UINT8_MAX)
static inline bool sa_ovf_sub_uint8_t(
		uint8_t minuend, uint8_t subtrahend, uint8_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#		elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#		else

	if ( minuend < subtrahend )
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#		endif
}
#	endif /* UINT8_MAX */

#	if defined(INT8_MAX) && defined(INT8_MIN)
static inline bool sa_ovf_sub_int8_t(
		int8_t minuend, int8_t subtrahend, int8_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#		elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#		else

	int_fast16_t temporary = (int_fast16_t) minuend - (int_fast16_t) subtrahend;

	if ( temporary > INT8_MAX || temporary < INT8_MIN )
		return true;
	*result_pointer = (int8_t) temporary;

	return false;

#		endif
}
#	endif /* INT8_MAX */

#	if defined(UINT16_MAX)
static inline bool sa_ovf_sub_uint16_t(
		uint16_t minuend, uint16_t subtrahend, uint16_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#		elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#		else

	if ( minuend < subtrahend )
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#		endif
}
#	endif /* UINT16_MAX */

#	if defined(INT16_MAX) && defined(INT16_MIN)
static inline bool sa_ovf_sub_int16_t(
		int16_t minuend, int16_t subtrahend, int16_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#		elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#		else

	int_fast32_t temporary = (int_fast32_t) minuend - (int_fast32_t) subtrahend;

	if ( temporary > INT16_MAX || temporary < INT16_MIN )
		return true;
	*result_pointer = (int16_t) temporary;

	return false;

#		endif
}
#	endif /* INT16_MAX */

#	if defined(UINT32_MAX)
static inline bool sa_ovf_sub_uint32_t(
		uint32_t minuend, uint32_t subtrahend, uint32_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#		elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#		else

	if ( minuend < subtrahend )
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#		endif
}
#	endif /* UINT32_MAX */

#	if defined(INT32_MAX) && defined(INT32_MIN)
static inline bool sa_ovf_sub_int32_t(
		int32_t minuend, int32_t subtrahend, int32_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#		elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#		else

	int_fast64_t temporary = (int_fast64_t) minuend - (int_fast64_t) subtrahend;

	if ( temporary > INT32_MAX || temporary < INT32_MIN )
		return true;
	*result_pointer = (int32_t) temporary;

	return false;

#		endif
}
#	endif /* INT32_MAX */

#	if defined(UINT64_MAX)
static inline bool sa_ovf_sub_uint64_t(
		uint64_t minuend, uint64_t subtrahend, uint64_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#		elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#		else

	if ( minuend < subtrahend )
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#		endif
}
#	endif /* UINT64_MAX */

#	if defined(UINT64_MAX) && defined(INT64_MAX)
static inline bool sa_ovf_sub_int64_t(
		int64_t minuend, int64_t subtrahend, int64_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#		elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#		else

	if ((minuend >= 0 && subtrahend < minuend - INT64_MAX) ||
		(minuend < 0  && subtrahend > minuend - INT64_MIN))
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#		endif
}
#	endif /* UINT64_MAX */

static inline bool sa_ovf_sub_size_t(
		size_t minuend, size_t subtrahend, size_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#	elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#	else

	if (minuend < subtrahend)
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#	endif
}

static inline bool sa_ovf_sub_ptrdiff_t(
		ptrdiff_t minuend, ptrdiff_t subtrahend, ptrdiff_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#	elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#	else

	if ((minuend >= 0 && subtrahend < minuend - PTRDIFF_MAX) ||
		(minuend < 0  && subtrahend > minuend - PTRDIFF_MIN))
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#	endif
}

static inline bool sa_ovf_sub_uintmax_t(
		uintmax_t minuend, uintmax_t subtrahend, uintmax_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#	elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#	else

	if (minuend < subtrahend)
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#	endif
}

static inline bool sa_ovf_sub_intmax_t(
		intmax_t minuend, intmax_t subtrahend, intmax_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#	elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#	else

	if ((minuend >= 0 && subtrahend < minuend - INTMAX_MAX) ||
		(minuend < 0  && subtrahend > minuend - INTMAX_MIN))
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#	endif
}

#	ifdef UINTPTR_MAX
static inline bool sa_ovf_sub_uintptr_t(
		uintptr_t minuend, uintptr_t subtrahend, uintptr_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#		elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#		else

	if (minuend < subtrahend )
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#		endif
}
#	endif /* UINTPTR_MAX */

#	ifdef INTPTR_MAX
static inline bool sa_ovf_sub_intptr_t(
		intptr_t minuend, intptr_t subtrahend, intptr_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_SUBTRACTION_STANDARD == 1

	return ckd_sub( result_pointer, minuend, subtrahend );

#		elif SA_SAFE_HAS_OVERFLOW_SUBTRACTION_BUILTIN == 1

	return __builtin_sub_overflow( minuend, subtrahend, result_pointer );

#		else

	if ((minuend >= 0 && subtrahend < minuend - INTPTR_MAX) ||
		(minuend < 0  && subtrahend > minuend - INTPTR_MIN))
		return true;

	*result_pointer = minuend - subtrahend;
	return false;

#		endif
}
#	endif /* INTPTR_MAX */

#endif /* SAFE_SUBTRACTION_H */
