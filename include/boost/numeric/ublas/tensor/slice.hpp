//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//
/// \file strides.hpp Definition for the basic_strides template class

#ifndef _BOOST_UBLAS_TENSOR_SLICE_HPP_
#define _BOOST_UBLAS_TENSOR_SLICE_HPP_

namespace boost::numeric::ublas::span{
  
    template<typename T, T...>
    struct basic_slice;

    template<typename T>
    inline static constexpr T const basic_slice_end = std::numeric_limits<T>::max();

    namespace detail
    {
        template<typename T, T val>
        constexpr T tensor_abs() noexcept{
            if constexpr( std::numeric_limits<T>::is_signed ){
                return val < T(0) ? -val : val;
            }else{
                return val;
            }
        }

        template<typename T>
        constexpr T tensor_abs(T val) noexcept{
            if constexpr( std::numeric_limits<T>::is_signed ){
                return val < T(0) ? -val : val;
            }else{
                return val;
            }
        }

    } // namespace detail
    

} // namespace boost::numeric::ublas::span

namespace boost::numeric::ublas::span{
  
    template<typename T, T Start, T End, T Stride, T Size>
    struct basic_slice<T, Start, End, Stride, Size>{
        
        using self_type = basic_slice<T, Start, End, Stride, Size>;
        using value_type = T;
        using size_type = std::size_t;
        static constexpr value_type const _start = Start;
        static constexpr value_type const _end = End;
        static constexpr value_type const _stride = Stride;
        static constexpr size_type const _size = static_cast<size_type>(Size);

        inline constexpr value_type start() const noexcept{
            return _start;
        }

        inline constexpr value_type end() const noexcept{
            return _end;
        }

        inline constexpr value_type stride() const noexcept{
            return _stride;
        }

        inline constexpr size_type size() const noexcept{
            return _size;
        }

        template<T... Values>
        inline constexpr bool operator==(basic_slice<T,Values...> const& rhs) noexcept{
            return start() == rhs.start() && 
                end() == rhs.end() &&
                stride() == rhs.stride();
        }

        template<T... Values>
        inline constexpr bool operator!=(basic_slice<T,Values...> const& rhs) noexcept{
            return !(*this == rhs);
        }
        
    };
  
    template<typename T, T Start, T End, T Stride>
    struct basic_slice<T, Start, End, Stride> 
        : basic_slice<T,Start, End, Stride, ( (End - Start) / detail::tensor_abs<T,Stride>() + 1)> {};
  
    template<typename T, T Start, T End>
    struct basic_slice<T, Start, End> 
        : basic_slice<T,Start, End, T(1)> {};
  
    template<typename T, T Start>
    struct basic_slice<T, Start> 
        : basic_slice<T,Start, basic_slice_end<T> > {};


    template<typename T>
    struct basic_slice<T> {

        using self_type = basic_slice<T>;
        using value_type = T;
        using size_type = std::size_t;

        constexpr basic_slice() = default;
        
        constexpr basic_slice(value_type start, value_type end, value_type stride, value_type size) 
            : m_start(start)
            , m_end(end)
            , m_stride(stride)
            , m_size(size)
        {}
        
        constexpr basic_slice(value_type start, value_type end = basic_slice_end<T>, value_type stride = value_type{1}) 
            : m_start(start)
            , m_end(end)
            , m_stride(stride)
            , m_size( ( end - start ) / detail::tensor_abs(stride) + 1ul)
        {}

        constexpr basic_slice(basic_slice const&) = default;
        constexpr basic_slice(basic_slice&&) = default;

        constexpr basic_slice& operator=(basic_slice const&) = default;
        constexpr basic_slice& operator=(basic_slice&&) = default;

        inline constexpr value_type start() const noexcept{
            return m_start;
        }

        inline constexpr value_type end() const noexcept{
            return m_end;
        }

        inline constexpr value_type stride() const noexcept{
            return m_stride;
        }

        inline constexpr size_type size() const noexcept{
            return m_size;
        }

        inline constexpr bool empty() const noexcept{
            return m_size == 0;
        }

        template<T... Values>
        inline constexpr bool operator==(basic_slice<T,Values...> const& rhs) noexcept{
            return start() == rhs.start() && 
                end() == rhs.end() &&
                stride() == rhs.stride();
        }

        template<T... Values>
        inline constexpr bool operator!=(basic_slice<T,Values...> const& rhs) noexcept{
            return !(*this == rhs);
        }

    private:
        value_type  m_start{};
        value_type  m_end{};
        value_type  m_stride{};
        size_type   m_size{};
    };

} // namespace boost::numeric::ublas::span

namespace boost::numeric::ublas::span{
    
    inline static constexpr auto const end = basic_slice_end<std::size_t>;

    template<std::size_t... Vs>
    using slice = basic_slice<std::size_t, Vs...>;


} // namespace boost::numeric::ublas::span


#include <boost/numeric/ublas/tensor/slice_detail/slice_helper.hpp>

#endif
