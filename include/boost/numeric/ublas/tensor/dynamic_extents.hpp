//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_DYNAMIC_EXTENTS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_DYNAMIC_EXTENTS_HPP

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>
#include <boost/numeric/ublas/tensor/type_traits.hpp>
#include <boost/numeric/ublas/tensor/detail/extents_functions.hpp>

namespace boost::numeric::ublas {

/** @brief Template class for storing tensor extents with runtime variable size.
 *
 * Proxy template class of std::vector<ExtentType>.
 *
 */
template<typename ExtentType>
struct basic_extents
{
    using base_type         = std::vector<ExtentType>;
    using value_type        = typename base_type::value_type;
    using const_reference   = typename base_type::const_reference;
    using reference         = typename base_type::reference;
    using size_type         = typename base_type::size_type;
    using const_pointer     = typename base_type::const_pointer;
    using const_iterator    = typename base_type::const_iterator;

    static_assert( std::numeric_limits<value_type>::is_integer, "Static error in basic_layout: type must be of type integer.");
    static_assert(!std::numeric_limits<value_type>::is_signed,  "Static error in basic_layout: type must be of type unsigned integer.");

    /** @brief Default constructs basic_extents
     *
     * @code auto ex = basic_extents<unsigned>{};
     */
    basic_extents()
      : _base{}
    {
    }

    /** @brief Copy constructs basic_extents from a one-dimensional container
     *
     * @code auto ex = basic_extents<unsigned>(  std::vector<unsigned>(3u,3u) );
     *
     * @note checks if size > 1 and all elements > 0
     *
     * @param b one-dimensional std::vector<ExtentType> container
     */
    explicit basic_extents(base_type const& b)
      : _base(b)
    {
        if ( !valid(*this) ){
            throw std::length_error("Error in basic_extents::basic_extents() : shape tuple is not a valid permutation: has zero elements.");
        }
    }

    /** @brief Move constructs basic_extents from a one-dimensional container
     *
     * @code auto ex = basic_extents<unsigned>(  std::vector<unsigned>(3u,3u) );
     *
     * @note checks if size > 1 and all elements > 0
     *
     * @param b one-dimensional container of type std::vector<ExtentType>
     */
    explicit basic_extents(base_type && b)
      : _base(std::move(b))
    {
        if (!valid(*this)){
            throw std::length_error("Error in basic_extents::basic_extents() : shape tuple is not a valid permutation: has zero elements.");
        }
    }

    /** @brief Constructs basic_extents from an initializer list
     *
     * @code auto ex = basic_extents<unsigned>{3,2,4};
     *
     * @note checks if size > 1 and all elements > 0
     *
     * @param l one-dimensional list of type std::initializer<ExtentType>
     */
    basic_extents(std::initializer_list<value_type> l)
      : basic_extents( base_type(std::move(l)) )
    {
    }

    /** @brief Constructs basic_extents from a range specified by two iterators
     *
     * @code auto ex = basic_extents<unsigned>(a.begin(), a.end());
     *
     * @note checks if size > 1 and all elements > 0
     *
     * @param first iterator pointing to the first element
     * @param last iterator pointing to the next position after the last element
     */
    basic_extents(const_iterator first, const_iterator last)
      : basic_extents ( base_type( first,last ) )
    {
    }

    /** @brief Copy constructs basic_extents */
    basic_extents(basic_extents const& l )
      : _base(l._base)
    {
    }

    /** @brief Move constructs basic_extents */
    basic_extents(basic_extents && l ) noexcept
      : _base(std::move(l._base))
    {
    }

    template<typename OtherExtents,
        typename = std::enable_if_t< is_extents_v<OtherExtents> >
    >
    basic_extents(OtherExtents const& e)
        : _base(e.size())
    {
        static_assert( is_extents_v<OtherExtents>, "boost::numeric::ublas::basic_fixed_rank_extents(OtherExtents const&) : " 
            "OtherExtents is not a tensor extents"
        );
        std::copy(e.begin(),e.end(), _base.begin());
    }

    ~basic_extents() = default;

    basic_extents& operator=(basic_extents other) noexcept
    {
        swap (*this, other);
        return *this;
    }

    friend void swap(basic_extents& lhs, basic_extents& rhs) noexcept{
        std::swap(lhs._base   , rhs._base   );
    }

    [[nodiscard]] inline
    constexpr const_pointer data() const noexcept
    {
        return this->_base.data();
    }

    [[nodiscard]] inline
    constexpr const_reference operator[] (size_type p) const
    {
        return this->_base[p];
    }

    [[nodiscard]] inline
    constexpr const_reference at (size_type p) const
    {
        return this->_base.at(p);
    }

    [[nodiscard]] inline
    constexpr reference operator[] (size_type p)
    {
        return this->_base[p];
    }

    [[nodiscard]] inline
    constexpr reference at (size_type p)
    {
        return this->_base.at(p);
    }

    [[nodiscard]] inline
    constexpr const_reference back () const
    {
        return this->_base.back();
    }


    [[nodiscard]] inline
    constexpr bool empty() const noexcept
    {
        return this->_base.empty();
    }

    [[nodiscard]] inline
    constexpr size_type size() const noexcept
    {
        return this->_base.size();
    }

    inline
    constexpr void clear() noexcept
    {
        this->_base.clear();
    }

    [[nodiscard]] inline
    constexpr const_iterator
    begin() const noexcept
    {
        return _base.begin();
    }

    [[nodiscard]] inline
    constexpr const_iterator
    end() const noexcept
    {
        return _base.end();
    }

    [[nodiscard]] inline
    constexpr base_type const& base() const noexcept { return _base; }

private:
    base_type _base;

};

} // namespace boost::numeric::ublas


#endif
