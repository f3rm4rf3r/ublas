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


/// \file tensor_core.hpp Definition for the tensor template class

#ifndef BOOST_UBLAS_tensor_core_IMPL_HPP
#define BOOST_UBLAS_tensor_core_IMPL_HPP

#include <initializer_list>

#include <boost/numeric/ublas/tensor/algorithms.hpp>
#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/expression_evaluation.hpp>
#include <boost/numeric/ublas/tensor/fixed_rank_extents.hpp>
#include <boost/numeric/ublas/tensor/static_extents.hpp>
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include <boost/numeric/ublas/tensor/index.hpp>
#include <boost/numeric/ublas/tensor/type_traits.hpp>
#include <boost/numeric/ublas/tensor/tags.hpp>

namespace boost::numeric::ublas {

template< class T >
class tensor_core:
        public detail::tensor_expression< tensor_core<T>,tensor_core<T> >
{

    using self_type                 = tensor_core<T>;

public:
    using tensor_traits             = T;

    template<class derived_type>
    using tensor_expression_type    = detail::tensor_expression<self_type,derived_type>;

    template<class derived_type>
    using matrix_expression_type    = matrix_expression<derived_type>;

    template<class derived_type>
    using vector_expression_type    = vector_expression<derived_type>;

    using super_type                = tensor_expression_type<self_type>;
    using storage_traits            = typename tensor_traits::storage_traits;

    using array_type                = typename storage_traits::array_type;
    using layout_type               = typename tensor_traits::layout_type;


    using size_type                 = typename storage_traits::size_type;
    using difference_type           = typename storage_traits::difference_type;
    using value_type                = typename storage_traits::value_type;

    using reference                 = typename storage_traits::reference;
    using const_reference           = typename storage_traits::const_reference;

    using pointer                   = typename storage_traits::pointer;
    using const_pointer             = typename storage_traits::const_pointer;

    using iterator                  = typename storage_traits::iterator;
    using const_iterator            = typename storage_traits::const_iterator;

    using reverse_iterator          = typename storage_traits::reverse_iterator;
    using const_reverse_iterator    = typename storage_traits::const_reverse_iterator;

    using tensor_temporary_type     = self_type;
    using storage_category          = dense_tag;
    using container_tag             = typename storage_traits::container_tag;
    using resizable_tag             = typename storage_traits::resizable_tag;

    using extents_type              = typename tensor_traits::extents_type;
    using strides_type              = typename tensor_traits::strides_type;

    using matrix_type               = matrix<value_type,layout_type, std::vector<value_type> >;
    using vector_type               = vector<value_type, std::vector<value_type> >;


    static_assert( std::is_same<layout_type,first_order>::value || 
                   std::is_same<layout_type,last_order >::value, 
                   "boost::numeric::tensor_core template class only supports first- or last-order storage formats.");
    
    /** @brief Constructs a tensor_core.
     *
     * @note the tensor_core is empty.
     * @note the tensor_core needs to reshaped for further use.
     *
     */
    inline
    constexpr tensor_core ()
    {
        if constexpr( is_static_v<extents_type> ){
            auto temp = tensor_core(extents_type{},resizable_tag{});
            swap(*this,temp);
        }
    }

    constexpr tensor_core( extents_type e, storage_resizable_container_tag )
        : tensor_expression_type<self_type>()
        , extents_(std::move(e))
        , strides_(extents_)
        , size_(product(extents_))
        , data_( size_ )
    {}

    constexpr tensor_core( extents_type e, storage_static_container_tag )
        : tensor_expression_type<self_type>()
        , extents_(std::move(e))
        , strides_(extents_)
        , size_(product(extents_))
    {
        if ( data_.size() < size_ ){
            throw std::length_error("boost::numeric::ublas::tensor_core(extents_type const&, tag::static_storage): "
                "size of requested storage exceeds the current container size"
            );
        }
    }

    /** @brief Constructs a tensor_core with an initializer list for dynamic_extents
     *
     * By default, its elements are initialized to 0.
     *
     * @code tensor_core<float> A{4,2,3}; @endcode
     *
     * @param l initializer list for setting the dimension extents of the tensor_core
     */
    explicit inline
    tensor_core (std::initializer_list<size_type> l)
        : tensor_core( std::move(l), resizable_tag{} )
    {}

    /** @brief Constructs a tensor_core with a \c shape
     *
     * By default, its elements are initialized to 0.
     *
     * @code tensor_core<float> A{extents{4,2,3}}; @endcode
     *
     * @param s initial tensor_core dimension extents
     */
    explicit inline
    tensor_core (extents_type const& s)
        : tensor_core( s, resizable_tag{} )
    {}

    /** @brief Constructs a tensor_core with a \c shape
     *
     * By default, its elements are initialized to 0.
     *
     * @code tensor_core<float> A{extents{4,2,3}}; @endcode
     *
     * @param s initial tensor_core dimension extents
     * @param i initial tensor_core with this value
     */
    explicit inline
    tensor_core (extents_type const& s, value_type const& i)
        : tensor_core( s, resizable_tag{} )
    {
        std::fill_n(begin(),size_,i);
    }

    /** @brief Constructs a tensor_core with a \c shape and initiates it with one-dimensional data
     *
     * @code tensor_core<float> A{extents{4,2,3}, array }; @endcode
     *
     *
     *  @param s initial tensor_core dimension extents
     *  @param a container of \c array_type that is copied according to the storage layout
     */
    inline
    tensor_core (extents_type const& s, const array_type &a)
        : tensor_core( s, resizable_tag{} )
    {
        if( size_ != a.size() ){
            throw std::runtime_error("boost::numeric::ublas::tensor_core(extents_type,array_type): array size mismatch with extents");
        }
        std::copy_n(a.begin(),size_,begin());
    }


    /** @brief Constructs a tensor_core with another tensor_core with a different layout
     *
     * @param other tensor_core with a different layout to be copied.
     */
    template<typename OtherTensor>
    tensor_core (const tensor_core<OtherTensor> &other)
        : tensor_core( other.extents(), resizable_tag{} )
    { 
        copy(this->rank(), this->extents().data(),
                this->data(), this->strides().data(),
                other.data(), other.strides().data());
        
    }


    /** @brief Constructs a tensor_core with an tensor_core expression
     *
     * @code tensor_core<float> A = B + 3 * C; @endcode
     *
     * @note type must be specified of tensor_core must be specified.
     * @note dimension extents are extracted from tensors within the expression.
     *
     * @param expr tensor_core expression
     * @param size tensor_core expression
     */
    template<typename other_tensor,typename derived_type>
    tensor_core (const detail::tensor_expression<other_tensor,derived_type> &expr)
        : tensor_core( detail::retrieve_extents(expr), resizable_tag{} )
    {
        static_assert(is_valid_tensor_v<other_tensor>,
            "boost::numeric::ublas::tensor_core(tensor_expression<other_tensor, derived_type> const&) : "
            "other_tensor should be a valid tensor type"
        );
        
        static_assert(std::is_same_v<value_type, typename other_tensor::value_type>,
            "boost::numeric::ublas::tensor_core(tensor_expression<other_tensor, derived_type> const&) : "
            "tensor_core and other_tensor should have same value type"
        );

        detail::eval( *this, expr );
    }

    constexpr tensor_core( matrix_type const& v )
        : tensor_core( basic_extents<std::size_t>{v.size1(), v.size2()}, resizable_tag{} )
    {
        if( extents_.size() != 2ul ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(const matrix &v)"
                " : order of extents are not correct, please check!"
            );
        }

        if( extents_[0] != v.size1() || extents_[1] != v.size2() ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(const matrix &v)"
                " : please check the extents it is not set properly, "
                "if extents is static please set the extents while specifying type"
            );
        }

        std::copy(v.data().begin(), v.data().end(), data_.begin());
    }

    constexpr tensor_core( matrix_type && v )
        : tensor_core( basic_extents<std::size_t>{v.size1(), v.size2()}, resizable_tag{} )
    {
        if( extents_.size() != 2ul ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(matrix &&v)"
                " : order of extents are not correct, please check!"
            );
        }
        
        if( extents_[0] != v.size1() || extents_[1] != v.size2() ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(matrix &&v)"
                " : please check the extents it is not set properly, "
                "if extents is static please set the extents while specifying type"
            );
        }

        std::move(v.data().begin(), v.data().end(),data_.begin());
    }

    constexpr tensor_core (const vector_type &v)
        : tensor_core( basic_extents<std::size_t>{ v.size(), typename extents_type::value_type{1} }, resizable_tag{} )
    {
        if( extents_.size() != 2ul ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(const vector_type &v)"
                " : order of extents are not correct, please check!"
            );
        }
        if( extents_[0] != v.size() || extents_[1] != 1ul ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(const vector_type &v)"
                " : please check the extents it is not set properly, "
                "if extents is static please set the extents while specifying type"
            );
        }

        std::copy(v.data().begin(), v.data().end(), data_.begin());
        
    }

    constexpr tensor_core (vector_type &&v)
        : tensor_core( basic_extents<std::size_t>{ v.size(), typename extents_type::value_type{1} }, resizable_tag{} )
    {
        
        if( extents_.size() != 2ul ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(vector_type &&v)"
                " : order of extents are not correct, please check!"
            );
        }
        if( extents_[0] != v.size() || extents_[1] != 1ul ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(vector_type &&v)"
                " : please check the extents it is not set properly, "
                "if extents is static please set the extents while specifying type"
            );
        }

        std::move(v.data().begin(), v.data().end(),data_.begin());
        
    }

    /** @brief Constructs a tensor_core with a matrix expression
     *
     * @code tensor_core<float> A = B + 3 * C; @endcode
     *
     * @note matrix expression is evaluated and pushed into a temporary matrix before assignment.
     * @note extents are automatically extracted from the temporary matrix
     *
     * @param expr matrix expression
     */
    template<class derived_type>
    tensor_core (const matrix_expression_type<derived_type> &expr)
        : tensor_core(  matrix_type ( expr )  )
    {
    }

    /** @brief Constructs a tensor_core with a vector expression
     *
     * @code tensor_core<float> A = b + 3 * b; @endcode
     *
     * @note matrix expression is evaluated and pushed into a temporary matrix before assignment.
     * @note extents are automatically extracted from the temporary matrix
     *
     * @param expr vector expression
     */
    template<class derived_type>
    tensor_core (const vector_expression_type<derived_type> &expr)
        : tensor_core(  vector_type ( expr )  )
    {
    }


    /** @brief Constructs a tensor_core from another tensor_core
     *
     *  @param v tensor_core to be copied.
     */
    inline
    tensor_core (const tensor_core &v)
        : tensor_expression_type<self_type>()
        , extents_ (v.extents_)
        , strides_ (v.strides_)
        , size_    (v.size_)
        , data_    (v.data_   )
    {}



    /** @brief Constructs a tensor_core from another tensor_core
     *
     *  @param v tensor_core to be moved.
     */
    inline
    tensor_core (tensor_core &&v)
        : tensor_expression_type<self_type>() //tensor_container<self_type> ()
        , extents_ (std::move(v.extents_))
        , strides_ (std::move(v.strides_))
        , size_    (std::move(v.size_))
        , data_    (std::move(v.data_   ))
    {}

    /** @brief Evaluates the tensor_expression and assigns the results to the tensor_core
     *
     * @code A = B + C * 2;  @endcode
     *
     * @note rank and dimension extents of the tensors in the expressions must conform with this tensor_core.
     *
     * @param expr expression that is evaluated.
     */
    template<class derived_type>
    tensor_core &operator = (const tensor_expression_type<derived_type> &expr)
    {
        detail::eval(*this, expr);
        return *this;
    }

    tensor_core& operator=(tensor_core other)
    {
        swap (*this, other);
        return *this;
    }

    tensor_core& operator=(const_reference v)
    {
        std::fill_n(this->begin(), size_, v);
        return *this;
    }

    /** @brief Returns true if the tensor_core is empty (\c size==0) */
    [[nodiscard]] inline
    constexpr bool empty () const noexcept{
        return this->data_.empty();
    }

    /** @brief Returns the size of the tensor_core */
    [[nodiscard]] inline
    constexpr size_type size () const noexcept{
        return this->size_;
    }

    /** @brief Returns the upper bound or max size of the tensor_core */
    [[nodiscard]] inline
    constexpr size_type max_size () const noexcept{
        return this->data_.size();
    }

    /** @brief Returns the size of the tensor_core */
    [[nodiscard]] inline
    constexpr size_type size (size_type r) const {
        return this->extents_.at(r);
    }

    /** @brief Returns the number of dimensions/modes of the tensor_core */
    [[nodiscard]] inline
    constexpr size_type rank () const noexcept{
        return this->extents_.size();
    }

    /** @brief Returns the number of dimensions/modes of the tensor_core */
    [[nodiscard]] inline
    constexpr size_type order () const noexcept{
        return this->extents_.size();
    }

    /** @brief Returns the strides of the tensor_core */
    [[nodiscard]] inline
    constexpr strides_type const& strides () const noexcept{
        return this->strides_;
    }

    /** @brief Returns the extents of the tensor_core */
    [[nodiscard]] inline
    constexpr extents_type const& extents () const noexcept{
        return this->extents_;
    }

    /** @brief Returns the strides of the tensor_core */
    [[nodiscard]] inline
    constexpr strides_type& strides () noexcept{
        return this->strides_;
    }

    /** @brief Returns the extents of the tensor_core */
    [[nodiscard]] inline
    constexpr extents_type& extents () noexcept{
        return this->extents_;
    }
    
    inline
    constexpr void invalidate_size(){
        this->size_ = product(this->extents_);
    }

    /** @brief Returns a \c const reference to the container. */
    [[nodiscard]] inline
    constexpr const_pointer data () const noexcept{
        return this->data_.data();
    }

    /** @brief Returns a \c const reference to the container. */
    [[nodiscard]] inline
    constexpr pointer data () noexcept{
        return this->data_.data();
    }

    /** @brief Returns a \c const reference to the underlying container. */
    [[nodiscard]] inline
    constexpr array_type const& base () const noexcept{
        return data_;
    }

    /** @brief Returns a reference to the underlying container. */
    [[nodiscard]] inline
    constexpr array_type& base () noexcept{
        return data_;
    }

    /** @brief Element access using a single index.
     *
     *  @code auto a = A[i]; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
    [[nodiscard]] inline
    constexpr const_reference operator [] (size_type i) const {
        return this->data_[i];
    }

    /** @brief Element access using a single index.
     *
     *  @code auto a = A[i]; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
    [[nodiscard]] inline
    constexpr reference operator [] (size_type i) {
        return this->data_[i];
    }

    /** @brief Element access using a multi-index or single-index.
     *
     *
     *  @code auto a = A.at(i,j,k); @endcode or
     *  @code auto a = A.at(i);     @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<class ... size_types>
    [[nodiscard]] inline
    constexpr const_reference at (size_type i, size_types ... is) const {
        if constexpr (sizeof...(is) == 0)
            return this->data_[i];
        else
            return this->data_[detail::access<0ul>(size_type(0),this->strides_,i,std::forward<size_types>(is)...)];
    }

    /** @brief Element access using a multi-index or single-index.
     *
     *
     *  @code A.at(i,j,k) = a; @endcode or
     *  @code A.at(i) = a;     @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<class ... size_types>
    [[nodiscard]] inline
    constexpr reference at (size_type i, size_types ... is) {
        if constexpr (sizeof...(is) == 0)
            return this->data_[i];
        else{
            auto temp = detail::access<0ul>(size_type(0),this->strides_,i,std::forward<size_types>(is)...);
            return this->data_[temp];
        }
    }

    /** @brief Element access using a single index.
     *
     *
     *  @code A(i) = a; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
    [[nodiscard]] inline
    constexpr const_reference operator()(size_type i) const {
        return this->data_[i];
    }


    /** @brief Element access using a single index.
     *
     *  @code A(i) = a; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
    [[nodiscard]] inline
    constexpr reference operator()(size_type i){
        return this->data_[i];
    }

    /** @brief Generates a tensor_core index for tensor_core contraction
     *
     *
     *  @code auto Ai = A(_i,_j,k); @endcode
     *
     *  @param i placeholder
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<std::size_t I, class ... index_types>
    [[nodiscard]] inline
    constexpr decltype(auto) operator() (index::index_type<I> p, index_types ... ps) const
    {
        constexpr auto N = sizeof...(ps)+1;
        if( N != this->rank() )
            throw std::runtime_error("Error in boost::numeric::ublas::operator(): size of provided index_types does not match with the rank.");

        return std::make_pair( std::cref(*this),  std::make_tuple( p, std::forward<index_types>(ps)... ) );
    }

    friend void swap(tensor_core& lhs, tensor_core& rhs) noexcept{
        std::swap(lhs.data_   , rhs.data_   );
        std::swap(lhs.extents_, rhs.extents_);
        std::swap(lhs.strides_, rhs.strides_);
        std::swap(lhs.size_, rhs.size_);
    }


    /// \brief return an iterator on the first element of the tensor_core
    [[nodiscard]] inline
    constexpr const_iterator begin () const noexcept{
        return data_.begin ();
    }

    /// \brief return an iterator on the first element of the tensor_core
    [[nodiscard]] inline
    constexpr const_iterator cbegin () const noexcept{
        return data_.cbegin ();
    }

    /// \brief return an iterator after the last element of the tensor_core
    [[nodiscard]] inline
    constexpr const_iterator end () const noexcept{
        return data_.end();
    }

    /// \brief return an iterator after the last element of the tensor_core
    [[nodiscard]] inline
    constexpr const_iterator cend () const noexcept{
        return data_.cend ();
    }

    /// \brief Return an iterator on the first element of the tensor_core
    [[nodiscard]] inline
    constexpr iterator begin () noexcept{
        return data_.begin();
    }

    /// \brief Return an iterator at the end of the tensor_core
    [[nodiscard]] inline
    constexpr iterator end () noexcept{
        return data_.end();
    }

    /// \brief Return a const reverse iterator before the first element of the reversed tensor_core (i.e. end() of normal tensor_core)
    [[nodiscard]] inline
    constexpr const_reverse_iterator rbegin () const noexcept{
        return data_.rbegin();
    }

    /// \brief Return a const reverse iterator before the first element of the reversed tensor_core (i.e. end() of normal tensor_core)
    [[nodiscard]] inline
    constexpr const_reverse_iterator crbegin () const noexcept{
        return data_.crbegin();
    }

    /// \brief Return a const reverse iterator on the end of the reverse tensor_core (i.e. first element of the normal tensor_core)
    [[nodiscard]] inline
    constexpr const_reverse_iterator rend () const noexcept{
        return data_.rend();
    }

    /// \brief Return a const reverse iterator on the end of the reverse tensor_core (i.e. first element of the normal tensor_core)
    [[nodiscard]] inline
    constexpr const_reverse_iterator crend () const noexcept{
        return data_.crend();
    }

    /// \brief Return a const reverse iterator before the first element of the reversed tensor_core (i.e. end() of normal tensor_core)
    [[nodiscard]] inline
    constexpr reverse_iterator rbegin () noexcept{
        return data_.rbegin();
    }

    /// \brief Return a const reverse iterator on the end of the reverse tensor_core (i.e. first element of the normal tensor_core)
    [[nodiscard]] inline
    constexpr reverse_iterator rend () noexcept{
        return data_.rend();
    }

private:

    extents_type extents_;
    strides_type strides_;
    size_type  size_{};
    array_type data_;
};

} // namespaces

#endif
