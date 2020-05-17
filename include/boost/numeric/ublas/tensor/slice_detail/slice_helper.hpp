//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_SLICE_HELPER_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_SLICE_HELPER_HPP_

#include <limits>
#include <cstddef>
#include <exception>
#include <tuple>

namespace boost::numeric::ublas::span::detail{

    template<typename>  
    struct always_false : std::false_type{};

    template<typename T, T, T, T, std::size_t>
    struct basic_static_slice_helper;

    template<typename T>
    struct basic_dynamic_slice_helper;

} // namespace boost::numeric::ublas::span::detail

namespace boost::numeric::ublas::span::detail{
  
    template<typename T, T Start, T End, T Stride, std::size_t Size>
    struct basic_static_slice_helper{
        
        using type =  basic_slice<T,Start,End,Stride,static_cast<T>(Size)>;

        inline constexpr auto operator()() const noexcept{
            return type{};
        }

    };

    template<typename T>
    struct basic_dynamic_slice_helper{
        
        using type = basic_slice<T>;

        constexpr basic_dynamic_slice_helper() = default;
        constexpr basic_dynamic_slice_helper(T start, T end, T stride, T size)
            : m_slice(start,end,stride,size)
        {}

        inline constexpr auto operator()() const noexcept{
            return m_slice;
        }

    private:
        type m_slice{};
    };

    template<typename T, T Start, T End, T Stride, std::size_t Ex>
    inline constexpr auto signed_slice_normalizer() noexcept{
        constexpr std::size_t const size = ( (End - Start) / tensor_abs<T,Stride>() ) + 1ul;
        
        static_assert(always_false<T>::value, "boost::numeric::ublas::span::detail::signed_slice_normalizer() : not yet implemented");
    }

    template<typename T, T Start, T End, T Stride, std::size_t Ex>
    inline constexpr auto unsigned_slice_normalizer() noexcept{
        
        constexpr T const last = End > Ex ? Ex : End;

        constexpr std::size_t const size = ( (last - Start) / Stride ) + 1ul;
        constexpr std::size_t const remaining = ( (last - Start) % Stride );
        
        constexpr T const new_end = last - static_cast<T>(remaining);
        
        if constexpr( Start == End ){
            return basic_static_slice_helper<T,new_end,new_end,Stride,size>{}();
        }else{
            static_assert(Stride != 0, "boost::numeric::ublas::span::detail::unsigned_slice_normalizer() : stride cannot be zero");
            static_assert(Start < End, "boost::numeric::ublas::span::detail::unsigned_slice_normalizer() : Invalid range. Start cannot greater than End");
            
            return basic_static_slice_helper<T, Start, new_end, Stride, size >{}();
        }
    }

    template<typename T, T Start, T End, T Stride, std::size_t Ex>
    inline constexpr auto slice_normalizer() noexcept{
        if constexpr( std::numeric_limits<T>::is_signed ){
            return signed_slice_normalizer<T,Start,End,Stride,Ex>();
        }else{
            return unsigned_slice_normalizer<T,Start,End,Stride,Ex>();
        }
    }



    template<typename T>
    inline constexpr auto signed_slice_normalizer(T, T, T, std::size_t){
        static_assert(always_false<T>::value, "boost::numeric::ublas::span::detail::signed_slice_normalizer() : not yet implemented");
    }

    template<typename T>
    inline constexpr auto unsigned_slice_normalizer(T start, T end, T stride, std::size_t extent){
        
        T const last = end > extent ? extent : end;

        std::size_t const size = ( (last - start) / stride ) + 1ul;
        std::size_t const remaining = ( (last - start) % stride );
        
        T const new_end = last - static_cast<T>(remaining);
        
        if( start == end ){
            return basic_dynamic_slice_helper<T>{ new_end, new_end, stride, size }();
        }else{
            if ( stride == 0 ) 
                throw std::runtime_error( "boost::numeric::ublas::span::detail::unsigned_slice_normalizer() : stride cannot be zero");
            if ( start > end ) 
                throw std::runtime_error( "boost::numeric::ublas::span::detail::unsigned_slice_normalizer() : Invalid range. start cannot greater than end");
            
            return basic_dynamic_slice_helper<T>{ start, new_end, stride, size }();
        }
    }

    template<typename T>
    inline constexpr auto slice_normalizer(T start, T end, T stride, std::size_t extent){
        if constexpr( std::numeric_limits<T>::is_signed ){
            return signed_slice_normalizer<T>(start,end,stride,extent);
        }else{
            return unsigned_slice_normalizer<T>(start,end,stride,extent);
        }
    }

    template<typename... Ts>
    struct wrap_tuple{
        using type = std::tuple<Ts...>;
        static constexpr std::size_t const size = sizeof...(Ts);

        template<std::size_t N>
        using element = std::tuple_element_t<N,type>;
    };

    template<typename T, typename... Ts>
    constexpr auto push_front(T, wrap_tuple<Ts...>) -> wrap_tuple<T,Ts...>;

    template<typename T, typename... Ts>
    constexpr auto push_back(T, wrap_tuple<Ts...>) -> wrap_tuple<Ts...,T>;
    
    template<typename T, typename... Ts>
    constexpr auto pop_front(wrap_tuple<T, Ts...>) -> wrap_tuple<Ts...>;

} // namespace boost::numeric::ublas::span::detail


#endif
