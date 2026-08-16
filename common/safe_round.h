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

/* Portable safe rounding to a multiple.
 *
 * Every sa_ovf_round_<direction>_<type> function round value to the nearest multiple
 * of mathematical modulus.
 * All operations are performed without undefined and platform defined behavior.
 *
 * Precondition:
 * result_pointer	- isn't NULL
 * modulus			- positive value
 *
 * Arguments:
 * value_to_round	- value to round
 * modulus			- positive modulus
 * result_pointer	- pointer to store rounded value
 *
 * Returns:
 * true				- result pointer is NULL, modulus invalid value, overflow occured
 * false			- rounding completed successfully
 */

#pragma once

#ifndef SAFE_ROUND_H
#define SAFE_ROUND_H

#	include "assert_m.h"		/* assert_check_m		*/
#	include "safe_modulus.h"	/* sa_math_mod_int64_t	*/
#	include "safe_addition.h"	/* sa_ovf_add_uint8_t	*/
#	include "safe_subtraction.h"/* sa_ovf_sub_int8_t	*/

#	include <stddef.h>	/* size_t	*/
#	include <stdint.h>	/* int8_t	*/
#	include <stdbool.h>	/* bool		*/

#	ifdef UINT64_MAX
static inline bool sa_ovf_round_up_uint64_t(
		uint64_t value_to_round, uint64_t modulus, uint64_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	uint64_t remainder = value_to_round % modulus;
	if ( remainder == 0 ) {
		*result_pointer = value_to_round;
		return false;
	}

	return sa_ovf_add_uint64_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_uint64_t(
		uint64_t value_to_round, uint64_t modulus, uint64_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus == 0 )
		return true;

	*result_pointer = value_to_round - (value_to_round % modulus);
	return false;
}
#	endif /* UINT64_MAX */

#	ifdef INT64_MAX
static inline bool sa_ovf_round_up_int64_t(
		int64_t value_to_round, int64_t modulus, int64_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus <= 0 )
		return true;

	int64_t remainder = 0;
	bool impossible_result = sa_math_mod_int64_t( value_to_round, modulus, &remainder );
	assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;
    if ( remainder == 0 ) {
		*result_pointer = value_to_round;
		return false;
	}
	return sa_ovf_add_int64_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_int64_t(
		int64_t value_to_round, int64_t modulus, int64_t * result_pointer
	)
{
	if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
		modulus <= 0 )
		return true;

	int64_t remainder = 0;
	bool impossible_result = sa_math_mod_int64_t( value_to_round, modulus, &remainder );
	assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;

	return sa_ovf_sub_int64_t( value_to_round, remainder, result_pointer );
}
#	endif /* INT64_MAX */

#   ifdef UINT32_MAX
static inline bool sa_ovf_round_up_uint32_t(
        uint32_t value_to_round, uint32_t modulus, uint32_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    uint32_t remainder = value_to_round % modulus;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }

    return sa_ovf_add_uint32_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_uint32_t(
        uint32_t value_to_round, uint32_t modulus, uint32_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    *result_pointer = value_to_round - (value_to_round % modulus);
    return false;
}
#   endif /* UINT32_MAX */

#   ifdef INT32_MAX
static inline bool sa_ovf_round_up_int32_t(
        int32_t value_to_round, int32_t modulus, int32_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    int32_t remainder = 0;
    bool impossible_result = sa_math_mod_int32_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }
    return sa_ovf_add_int32_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_int32_t(
        int32_t value_to_round, int32_t modulus, int32_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    int32_t remainder = 0;
    bool impossible_result = sa_math_mod_int32_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;

    return sa_ovf_sub_int32_t( value_to_round, remainder, result_pointer );
}
#   endif /* INT32_MAX */

#   ifdef UINT16_MAX
static inline bool sa_ovf_round_up_uint16_t(
        uint16_t value_to_round, uint16_t modulus, uint16_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    uint16_t remainder = value_to_round % modulus;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }

    return sa_ovf_add_uint16_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_uint16_t(
        uint16_t value_to_round, uint16_t modulus, uint16_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    *result_pointer = value_to_round - (value_to_round % modulus);
    return false;
}
#   endif /* UINT16_MAX */

#   ifdef INT16_MAX
static inline bool sa_ovf_round_up_int16_t(
        int16_t value_to_round, int16_t modulus, int16_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    int16_t remainder = 0;
    bool impossible_result = sa_math_mod_int16_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }
    return sa_ovf_add_int16_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_int16_t(
        int16_t value_to_round, int16_t modulus, int16_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    int16_t remainder = 0;
    bool impossible_result = sa_math_mod_int16_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;

    return sa_ovf_sub_int16_t( value_to_round, remainder, result_pointer );
}
#   endif /* INT16_MAX */

#   ifdef UINT8_MAX
static inline bool sa_ovf_round_up_uint8_t(
        uint8_t value_to_round, uint8_t modulus, uint8_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    uint8_t remainder = value_to_round % modulus;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }

    return sa_ovf_add_uint8_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_uint8_t(
        uint8_t value_to_round, uint8_t modulus, uint8_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    *result_pointer = value_to_round - (value_to_round % modulus);
    return false;
}
#   endif /* UINT8_MAX */

#   ifdef INT8_MAX
static inline bool sa_ovf_round_up_int8_t(
        int8_t value_to_round, int8_t modulus, int8_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    int8_t remainder = 0;
    bool impossible_result = sa_math_mod_int8_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }
    return sa_ovf_add_int8_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_int8_t(
        int8_t value_to_round, int8_t modulus, int8_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    int8_t remainder = 0;
    bool impossible_result = sa_math_mod_int8_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;

    return sa_ovf_sub_int8_t( value_to_round, remainder, result_pointer );
}
#   endif /* INT8_MAX */

static inline bool sa_ovf_round_up_uintmax_t(
        uintmax_t value_to_round, uintmax_t modulus, uintmax_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    uintmax_t remainder = value_to_round % modulus;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }

    return sa_ovf_add_uintmax_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_uintmax_t(
        uintmax_t value_to_round, uintmax_t modulus, uintmax_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    *result_pointer = value_to_round - (value_to_round % modulus);
    return false;
}

static inline bool sa_ovf_round_up_intmax_t(
        intmax_t value_to_round, intmax_t modulus, intmax_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    intmax_t remainder = 0;
    bool impossible_result = sa_math_mod_intmax_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }
    return sa_ovf_add_intmax_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_intmax_t(
        intmax_t value_to_round, intmax_t modulus, intmax_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    intmax_t remainder = 0;
    bool impossible_result = sa_math_mod_intmax_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;

    return sa_ovf_sub_intmax_t( value_to_round, remainder, result_pointer );
}

#   ifdef UINTPTR_MAX
static inline bool sa_ovf_round_up_uintptr_t(
        uintptr_t value_to_round, uintptr_t modulus, uintptr_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    uintptr_t remainder = value_to_round % modulus;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }

    return sa_ovf_add_uintptr_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_uintptr_t(
        uintptr_t value_to_round, uintptr_t modulus, uintptr_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    *result_pointer = value_to_round - (value_to_round % modulus);
    return false;
}
#   endif /* UINTPTR_MAX */

#   ifdef INTPTR_MAX
static inline bool sa_ovf_round_up_intptr_t(
        intptr_t value_to_round, intptr_t modulus, intptr_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    intptr_t remainder = 0;
    bool impossible_result = sa_math_mod_intptr_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }
    return sa_ovf_add_intptr_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_intptr_t(
        intptr_t value_to_round, intptr_t modulus, intptr_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    intptr_t remainder = 0;
    bool impossible_result = sa_math_mod_intptr_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;

    return sa_ovf_sub_intptr_t( value_to_round, remainder, result_pointer );
}
#   endif /* INTPTR_MAX */

static inline bool sa_ovf_round_up_size_t(
        size_t value_to_round, size_t modulus, size_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    size_t remainder = value_to_round % modulus;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }

    return sa_ovf_add_size_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_size_t(
        size_t value_to_round, size_t modulus, size_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus == 0 )
        return true;

    *result_pointer = value_to_round - (value_to_round % modulus);
    return false;
}

static inline bool sa_ovf_round_up_ptrdiff_t(
        ptrdiff_t value_to_round, ptrdiff_t modulus, ptrdiff_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    ptrdiff_t remainder = 0;
    bool impossible_result = sa_math_mod_ptrdiff_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;
    if ( remainder == 0 ) {
        *result_pointer = value_to_round;
        return false;
    }
    return sa_ovf_add_ptrdiff_t( value_to_round, modulus - remainder, result_pointer );
}

static inline bool sa_ovf_round_down_ptrdiff_t(
        ptrdiff_t value_to_round, ptrdiff_t modulus, ptrdiff_t * result_pointer
    )
{
    if( assert_check_m( result_pointer != NULL, "No place to store result found" ) == false ||
        modulus <= 0 )
        return true;

    ptrdiff_t remainder = 0;
    bool impossible_result = sa_math_mod_ptrdiff_t( value_to_round, modulus, &remainder );
    assert_m( impossible_result == false, "Impossible modulus overflow happened" );
    (void) impossible_result;

    return sa_ovf_sub_ptrdiff_t( value_to_round, remainder, result_pointer );
}

#endif /* SAFE_ROUND_H */
