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

/* Portable overflow-checked integer multiplication.
 *
 * Every sa_ovf_mul_<type> function multiplies two values of given type
 * without undefined behaviour, unlike the built-in '*' operator.
 *
 * Precondition:
 * result_pointer	- isn't NULL
 *
 * Arguments:
 * first			- multiplicand
 * second			- multiplier
 * result_pointer	- pointer to store the product
 *
 * Returns:
 * true				- overflow occurred, result pointer is NULL
 * false			- multiplication completed successfully
 */

#pragma once

#ifndef SAFE_MULTIPLICATION_H
#define SAFE_MULTIPLICATION_H

#	include "assert_m.h"/* assert_check_m */

#	include <stdint.h>	/* uint8_t	*/
#	include <stddef.h>	/* size_t	*/
#	include <stdbool.h>	/* bool		*/

#	ifndef __has_builtin
#		define __has_builtin(x) 0
#	endif

#	if defined(__STDC_VERSION__) && (__STDC_VERSION__ + 0L) >= 202311L &&\
		defined(__has_include) && __has_include(<stdckdint.h>)
#		define SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD 1
#		include <stdckdint.h>
#	else
#		define SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD 0
#	endif /* stdckdint.h */

#	if (__has_builtin(__builtin_mul_overflow) != 0)
#		define SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN 1
#	else
#		define SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN 0
#	endif /* __has_builtin(__builtin_mul_overflow) */

#	if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 0
#		if defined(__SIZEOF_INT128__) && (defined(__GNUC__) || defined(__clang__))
#			define SA_SAFE_MULTIPLICATION_HAS_INT128 1
	static_assert_m(
		sizeof(__int128) >= sizeof(uint64_t) * 2,
		"No integer type wide enough for 128 bits found"
	);
#		else
#			define SA_SAFE_MULTIPLICATION_HAS_INT128 0
#		endif /* __SIZEOF_INT128__ */
#	else
#		define SA_SAFE_MULTIPLICATION_HAS_INT128 0
#	endif /* SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN */

#	if defined(UINT8_MAX)
static inline bool sa_ovf_mul_uint8_t(
		uint8_t first, uint8_t second, uint8_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#		else

	uint_fast16_t temporary = (uint_fast16_t) first * (uint_fast16_t) second;
	if ( temporary > UINT8_MAX )
		return true;

	*result_pointer = (uint8_t) temporary;
	return false;

#		endif
}
#	endif /* UINT8_MAX */

#	if defined(INT8_MAX)
static inline bool sa_ovf_mul_int8_t(
		int8_t first, int8_t second, int8_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#		else

	int_fast16_t temporary = (int_fast16_t) first * (int_fast16_t) second;
	if ( temporary > INT8_MAX || temporary < INT8_MIN )
		return true;

	*result_pointer = (int8_t) temporary;
	return false;

#		endif
}
#	endif /* INT8_MAX */

#	if defined(UINT16_MAX)
static inline bool sa_ovf_mul_uint16_t(
		uint16_t first, uint16_t second, uint16_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#		else

	uint_fast32_t temporary = (uint_fast32_t) first * (uint_fast32_t) second;
	if ( temporary > UINT16_MAX )
		return true;

	*result_pointer = (uint16_t) temporary;
	return false;

#		endif
}
#	endif /* UINT16_MAX */

#	if defined(INT16_MAX)
static inline bool sa_ovf_mul_int16_t(
		int16_t first, int16_t second, int16_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#		else

	int_fast32_t temporary = (int_fast32_t) first * (int_fast32_t) second;
	if ( temporary > INT16_MAX || temporary < INT16_MIN)
		return true;

	*result_pointer = (int16_t) temporary;
	return false;

#		endif
}
#	endif /* INT16_MAX */

#	if defined(UINT32_MAX)
static inline bool sa_ovf_mul_uint32_t(
		uint32_t first, uint32_t second, uint32_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#		else

	uint_fast64_t temporary = (uint_fast64_t) first * (uint_fast64_t) second;
	if ( temporary > UINT32_MAX )
		return true;

	*result_pointer = (uint32_t) temporary;
	return false;

#		endif
}
#	endif /* UINT32_MAX */

#	if defined(INT32_MAX)
static inline bool sa_ovf_mul_int32_t(
		int32_t first, int32_t second, int32_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#		else

	int_fast64_t temporary = (int_fast64_t) first * (int_fast64_t) second;
	if ( temporary > INT32_MAX || temporary < INT32_MIN)
		return true;

	*result_pointer = (int32_t) temporary;
	return false;

#		endif
}
#	endif /* INT32_MAX */

#	if defined(UINT64_MAX)
static inline bool sa_ovf_mul_uint64_t(
		uint64_t first, uint64_t second, uint64_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#		elif SA_SAFE_MULTIPLICATION_HAS_INT128 == 1

	unsigned __int128 temporary = (unsigned __int128) first * (unsigned __int128) second;
	if ( temporary > UINT64_MAX )
		return true;

	*result_pointer = (uint64_t) temporary;
	return false;

#		else

	if (first != 0 && second > UINT64_MAX / first )
		return true;

	*result_pointer = first * second;
	return false;

#		endif
}
#	endif /* UINT64_MAX */

#	if defined(INT64_MAX)
static inline bool sa_ovf_mul_int64_t(
		int64_t first, int64_t second, int64_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#		elif SA_SAFE_MULTIPLICATION_HAS_INT128 == 1

	__int128 temporary = (__int128) first * (__int128) second;
	if ( temporary > INT64_MAX || temporary < INT64_MIN )
		return true;

	*result_pointer = (int64_t) temporary;
	return false;

#		else

	if ( first == 0 || second == 0 ) {
		*result_pointer = 0;
		return false;
	}

	uint64_t modulo_first	= (first	> 0) ? (uint64_t) first	: -(uint64_t) first;
	uint64_t modulo_second	= (second	> 0) ? (uint64_t) second: -(uint64_t) second;
	bool positive = (first > 0 && second > 0) || (first < 0 && second < 0);

	if ( positive == true && modulo_first > (uint64_t) INT64_MAX / modulo_second)	return true;
	else if ( modulo_first > ((uint64_t) INT64_MAX + 1) / modulo_second)			return true;

	*result_pointer = first * second;
	return false;

#		endif /* SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1 */
}
#	endif /* INT64_MAX */

static inline bool sa_ovf_mul_size_t(
		size_t first, size_t second, size_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#	elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#	elif SA_SAFE_MULTIPLICATION_HAS_INT128 == 1 &&\
		( defined(__SIZEOF_SIZE_T__) ? (__SIZEOF_SIZE_T__ <= 8) : (SIZE_MAX <= UINT64_MAX) )

	unsigned __int128 temporary = (unsigned __int128) first * (unsigned __int128) second;
	if ( temporary > SIZE_MAX )
		return true;

	*result_pointer = (size_t) temporary;
	return false;

#	else

	if (first != 0 && second > SIZE_MAX / first )
		return true;

	*result_pointer = first * second;
	return false;

#	endif
}

static inline bool sa_ovf_mul_ptrdiff_t(
		ptrdiff_t first, ptrdiff_t second, ptrdiff_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#	elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#	elif SA_SAFE_MULTIPLICATION_HAS_INT128 == 1 &&	\
		( defined(__SIZEOF_PTRDIFF_T__) ?			\
			(__SIZEOF_PTRDIFF_T__ <= 8) : (PTRDIFF_MAX <= INT64_MAX) )

	__int128 temporary = (__int128) first * (__int128) second;
	if ( temporary > PTRDIFF_MAX || temporary < PTRDIFF_MIN )
		return true;

	*result_pointer = (ptrdiff_t) temporary;
	return false;

#	else

	if ( first == 0 || second == 0 ) {
		*result_pointer = 0;
		return false;
	}

	uintmax_t modulo_first	= (first	> 0) ? (uintmax_t) first	: -(uintmax_t) first;
	uintmax_t modulo_second	= (second	> 0) ? (uintmax_t) second	: -(uintmax_t) second;
	bool positive = (first > 0 && second > 0) || (first < 0 && second < 0);

	if ( positive == true && modulo_first > (uintmax_t) PTRDIFF_MAX / modulo_second)return true;
	else if ( modulo_first > ((uintmax_t) PTRDIFF_MAX + 1) / modulo_second)			return true;

	*result_pointer = first * second;
	return false;

#	endif
}

static inline bool sa_ovf_mul_uintmax_t(
		uintmax_t first, uintmax_t second, uintmax_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#	elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#	elif SA_SAFE_MULTIPLICATION_HAS_INT128 == 1 && \
		(defined(__UINTMAX_WIDTH__) ? (__UINTMAX_WIDTH__ <= 64) : (UINTMAX_MAX <= UINT64_MAX) )

	unsigned __int128 temporary = (unsigned __int128) first * (unsigned __int128) second;
	if ( temporary > UINTMAX_MAX )
		return true;

	*result_pointer = (uintmax_t) temporary;
	return false;

#	else

	if (first != 0 && second > UINTMAX_MAX / first)
		return true;

	*result_pointer = first * second;
	return false;

#	endif
}

static inline bool sa_ovf_mul_intmax_t(
		intmax_t first, intmax_t second, intmax_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#	if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#	elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#	elif SA_SAFE_MULTIPLICATION_HAS_INT128 == 1 && \
		(defined(__INTMAX_WIDTH__) ? (__INTMAX_WIDTH__ <= 64) : (INTMAX_MAX <= INT64_MAX) )

	__int128 temporary = (__int128) first * (__int128) second;
	if ( temporary > INTMAX_MAX || temporary < INTMAX_MIN )
		return true;

	*result_pointer = (intmax_t) temporary;
	return false;

#	else

	if ( first == 0 || second == 0 ) {
		*result_pointer = 0;
		return false;
	}

	uintmax_t modulo_first	= (first	> 0) ? (uintmax_t) first	: -(uintmax_t) first;
	uintmax_t modulo_second	= (second	> 0) ? (uintmax_t) second	: -(uintmax_t) second;
	bool positive = (first > 0 && second > 0) || (first < 0 && second < 0);

	if ( positive == true && modulo_first > (uintmax_t) INTMAX_MAX / modulo_second)	return true;
	else if ( modulo_first > ((uintmax_t) INTMAX_MAX + 1) / modulo_second)			return true;

	*result_pointer = first * second;
	return false;

#	endif
}

#	ifdef UINTPTR_MAX
static inline bool sa_ovf_mul_uintptr_t(
		uintptr_t first, uintptr_t second, uintptr_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#		elif SA_SAFE_MULTIPLICATION_HAS_INT128 == 1 &&	\
			( defined(__UINTPTR_WIDTH__) ?				\
				(__UINTPTR_WIDTH__ <= 64) : (UINTPTR_MAX <= UINT64_MAX) )

	unsigned __int128 temporary = (unsigned __int128) first * (unsigned __int128) second;
	if ( temporary > UINTPTR_MAX )
		return true;

	*result_pointer = (uintptr_t) temporary;
	return false;

#		else

	if (first != 0 && second > UINTPTR_MAX / first)
		return true;

	*result_pointer = first * second;
	return false;

#		endif
}
#	endif /* UINTPTR_MAX */

#	ifdef INTPTR_MAX
static inline bool sa_ovf_mul_intptr_t(
		intptr_t first, intptr_t second, intptr_t * result_pointer
	)
{
	if ( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false )
		return true;

#		if SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_STANDARD == 1

	return ckd_mul( result_pointer, first, second );

#		elif SA_SAFE_HAS_OVERFLOW_MULTIPLICATION_BUILTIN == 1

	return __builtin_mul_overflow( first, second, result_pointer );

#		elif SA_SAFE_MULTIPLICATION_HAS_INT128 == 1 &&	\
			( defined(__INTPTR_WIDTH__) ?				\
				(__INTPTR_WIDTH__ <= 64) : (INTPTR_MAX <= INT64_MAX) )

	__int128 temporary = (__int128) first * (__int128) second;
	if ( temporary > INTPTR_MAX || temporary < INTPTR_MIN )
		return true;

	*result_pointer = (intptr_t) temporary;
	return false;

#		else

	if ( first == 0 || second == 0 ) {
		*result_pointer = 0;
		return false;
	}

	uintmax_t modulo_first	= (first	> 0) ? (uintmax_t) first	: -(uintmax_t) first;
	uintmax_t modulo_second	= (second	> 0) ? (uintmax_t) second	: -(uintmax_t) second;
	bool positive = (first > 0 && second > 0) || (first < 0 && second < 0);

	if ( positive == true && modulo_first > (uintmax_t) INTPTR_MAX / modulo_second)	return true;
	else if ( modulo_first > ((uintmax_t) INTPTR_MAX + 1) / modulo_second)			return true;

	*result_pointer = first * second;
	return false;

#		endif
}
#	endif /* INTPTR_MAX */

#endif /* SAFE_MULTIPLICATION_H */
