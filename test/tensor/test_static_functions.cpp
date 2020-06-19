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
//  And we acknowledge the support from all contributors.


#include <iostream>
#include <algorithm>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <boost/test/unit_test.hpp>

#include "utility.hpp"

// BOOST_AUTO_TEST_SUITE ( test_tensor_functions, * boost::unit_test::depends_on("test_tensor_contraction") )
BOOST_AUTO_TEST_SUITE ( test_tensor_functions)


using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::first_order>;


struct fixture
{
    template<size_t... E>
    using static_extents_type = boost::numeric::ublas::static_extents<E...>;
    
    using dynamic_extents_type = boost::numeric::ublas::extents<>;

    fixture()
      : extents {
          dynamic_extents_type{1,1}, // 1
          dynamic_extents_type{2,3}, // 2
          dynamic_extents_type{2,3,1}, // 3
          dynamic_extents_type{4,2,3}, // 4
          dynamic_extents_type{4,2,3,5}} // 5
    {
    }

    std::tuple<
        static_extents_type<1,1>, // 1
        static_extents_type<2,3>, // 2
        static_extents_type<2,3,1>, // 3
        static_extents_type<4,2,3>, // 4
        static_extents_type<4,2,3,5> // 5
    > static_extents{};


    std::vector<dynamic_extents_type> extents;
};



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_prod_vector, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    for_each_tuple(static_extents,[](auto const&, auto& n){                                   
        using extents_type = typename std::decay<decltype(n)>::type;              
        using tensor_type = ublas::static_tensor<value_type, extents_type, layout_type>; 
        using vector_type = typename tensor_type::vector_type;                    
        auto a = tensor_type(n,value_type{2});
        
        for (auto m = 0u; m < n.size(); ++m) {                                    
                                                                                
        auto b = vector_type(n[m], value_type{1});                               
                                                                                
        auto c = ublas::prod(a, b, m + 1);                                       
                                                                                
        for (auto i = 0u; i < c.size(); ++i)                                     
            BOOST_CHECK_EQUAL(c[i], value_type( static_cast< inner_type_t<value_type> >(n[m]) ) * a[i]);                     
        }                                   
    });

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_tensor_1, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    // left-hand and right-hand side have the
    // the same number of elements
    for_each_tuple(static_extents,[](auto, auto const& n){
        using extents_type = std::decay_t< decltype(n) >;
        using tensor_type = ublas::static_tensor<value_type,extents_type,layout_type>;

        auto a  = tensor_type( n, value_type{2} );
        auto b  = tensor_type( n, value_type{3} );

        constexpr auto const pa = extents_type::_size;

        // the number of contractions is changed.
        boost::mp11::mp_for_each< boost::mp11::mp_iota_c<pa + 1> >([&](auto iter){
            constexpr auto const q = decltype(iter)::value;
            using mp_phi = boost::mp11::mp_pop_front< boost::mp11::mp_iota_c< q + 1> >;
            using phi = mp_list_to_int_seq_t< mp_phi >;
            
            auto phi_array = ublas::detail::seq_to_array_v<phi>;

            auto c = ublas::prod(a, b, phi{});

            auto acc = value_type(1);
            for(auto i = 0ul; i < q; ++i)
                acc *= value_type( static_cast< inner_type_t<value_type> >( a.extents().at(phi_array.at(i)-1) ) );

            for(auto i = 0ul; i < c.size(); ++i)
                BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );

        });

    });
}

template<typename PiList>
struct test_tensor_prod_tensor_2_transform_fn{
    using pi = PiList;

    template<typename I>
    struct fn{
        using type = boost::mp11::mp_at_c<PiList, I::value - 1 >;
    };
};

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_tensor_2, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    // left-hand and right-hand side have the
    // the same number of elements
    for_each_tuple(static_extents,[](auto, auto const& n){
        using extents_type = std::decay_t< decltype(n) >;
        using tensor_type = ublas::static_tensor<value_type,extents_type,layout_type>;

        auto a  = tensor_type( n, value_type{2} );
        constexpr auto const pa = extents_type::_size;

        using mp_phi = boost::mp11::mp_pop_front< boost::mp11::mp_iota_c< pa + 1> >;

        constexpr auto const fac = static_factorial_v<pa>;

        ublas::detail::static_for<0,fac>([&](auto, auto perm){
            
            using permutated_pi = std::decay_t< decltype(perm) >;

            auto inverse_lm = [&](auto iter, auto na){
                using iter_type = decltype(iter);
                constexpr auto const J = iter_type::value;

                using nb_type = decltype(na);
                using pi_at = boost::mp11::mp_at<permutated_pi,iter_type>;
                using replace_val = std::integral_constant< typename extents_type::value_type, extents_type::at(J) >;
                using res_type = boost::mp11::mp_replace_at_c<nb_type, pi_at::value - 1, replace_val >;
                return res_type{};
            };

            auto mp_nb = ublas::detail::type_static_for<0,pa>(
                    std::move(inverse_lm), 
                    ublas::detail::static_extents_to_seq_t<extents_type>{}
                );
            using nb_type = ublas::detail::seq_to_static_extents_t< decltype(mp_nb) >;
            
            using tensorb_type = ublas::static_tensor<value_type,nb_type,layout_type>;

            auto b  = tensorb_type( nb_type{}, value_type{3} );

            // the number of contractions is changed.
            boost::mp11::mp_for_each< boost::mp11::mp_iota_c<pa + 1> >([&](auto iter){
                constexpr auto const q = decltype(iter)::value;

                using mp_phia = boost::mp11::mp_pop_front< boost::mp11::mp_iota_c< q + 1 > >;

                auto mp_res_phib = ublas::detail::type_static_for<0,q>([&](auto iter, auto prev){
                    constexpr auto const i = decltype(iter)::value;
                    using pi_at = boost::mp11::mp_at_c<permutated_pi,i>;
                    return boost::mp11::mp_push_back<decltype(prev),pi_at>{};
                },boost::mp11::mp_list<>{});

                using mp_phib = decltype(mp_res_phib);
                using phia = mp_list_to_int_seq_t<mp_phia>;
                using phib = mp_list_to_int_seq_t<mp_phib>;

                auto c = ublas::prod(a, b, phia{}, phib{});
                auto phia_arr = ublas::detail::seq_to_array_v<phia>;
                auto acc = value_type(1);
                for(auto i = 0ul; i < q; ++i)
                    acc *= value_type( static_cast< inner_type_t<value_type> >( a.extents().at(phia_arr.at(i)-1) ) );

                for(auto i = 0ul; i < c.size(); ++i)
                    BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );

            });
            return static_next_permutation< permutated_pi >();
        },mp_phi{});

    });
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_inner_prod, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    auto const body = [&](auto const& a, auto const& b){
        auto c = ublas::inner_prod(a, b);
        auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_type(0));

        BOOST_CHECK_EQUAL( c , r );
    };

    for_each_tuple(static_extents,[&](auto const&, auto & n){
        using extents_type_1 = typename std::decay<decltype(n)>::type;             
        using extents_type_2 = typename std::decay<decltype(n)>::type;             
        using tensor_type_1 = ublas::static_tensor<value_type, extents_type_1, layout_type>;
        using tensor_type_2 = ublas::static_tensor<value_type, extents_type_2, layout_type>;
        auto a  = tensor_type_1(n,value_type(2));
        auto b  = tensor_type_2(n,value_type(1));
        body(a,b);

    });

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_outer_prod, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    for_each_tuple(static_extents,[&](auto const&, auto const& n1){
        using extents_type_1 = typename std::decay<decltype(n1)>::type;             
        using tensor_type_1 = ublas::static_tensor<value_type, extents_type_1, layout_type>;
        auto a  = tensor_type_1(n1,value_type(2));
        for_each_tuple(static_extents,[&](auto const& J, auto const& n2){
            using extents_type_2 = typename std::decay<decltype(n2)>::type;             
            using tensor_type_2 = ublas::static_tensor<value_type, extents_type_2, layout_type>;
            auto b  = tensor_type_2(n2,value_type(1));
            auto c  = ublas::outer_prod(a, b);

            for(auto const& cc : c)
                BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
            
        });

    });

}


BOOST_FIXTURE_TEST_CASE( test_tensor_real_imag_conj, fixture )
{
    using namespace boost::numeric;
    using value_type   = float;
    using complex_type = std::complex<value_type>;
    using layout_type  = ublas::first_order;

    for_each_tuple(static_extents,[](auto, auto const& n){
        using extents_type = std::decay_t< decltype(n) >;
        using tensor_complex_type = ublas::static_tensor<complex_type,extents_type,layout_type>;
        using tensor_type = ublas::static_tensor<value_type,extents_type,layout_type>;

        {
            auto a   = tensor_type(n);
            auto r0  = tensor_type(n);
            auto r00 = tensor_complex_type(n);


            auto one = value_type(1);
            auto v = one;
            for(auto& aa: a)
                aa = v, v += one;

            tensor_type b = (a+a) / value_type( 2 );
            tensor_type r1 = ublas::real( (a+a) / value_type( 2 )  );
            std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
            BOOST_CHECK( (bool) (r0 == r1) );

            tensor_type r2 = ublas::imag( (a+a) / value_type( 2 )  );
            std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
            BOOST_CHECK( (bool) (r0 == r2) );
            
            tensor_complex_type r3 = ublas::conj( (a+a) / value_type( 2 )  );
            std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
        }

        {
            auto a   = tensor_complex_type(n);

            auto r00 = tensor_complex_type(n);
            auto r0  = tensor_type(n);

            auto one = complex_type(1,1);
            auto v = one;
            for(auto& aa: a)
                aa = v, v = v + one;

            tensor_complex_type b = (a+a) / complex_type( 2,2 );


            tensor_type r1 = ublas::real( (a+a) / complex_type( 2,2 )  );
            std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
            BOOST_CHECK( (bool) (r0 == r1) );

            tensor_type r2 = ublas::imag( (a+a) / complex_type( 2,2 )  );
            std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
            BOOST_CHECK( (bool) (r0 == r2) );

            tensor_complex_type r3 = ublas::conj( (a+a) / complex_type( 2,2 )  );
            std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
            BOOST_CHECK( (bool) (r00 == r3) );
        }

    });


}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_norm, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    for_each_tuple(static_extents,[](auto, auto const& n){
        using extents_type = std::decay_t< decltype(n) >;
        using tensor_type = ublas::static_tensor<value_type,extents_type,layout_type>;

        auto a  = tensor_type(n);

        auto one = value_type(1);
        auto v = one;
        for(auto& aa: a)
            aa = v, v += one;

        auto c = ublas::inner_prod(a, a);
        auto r = std::inner_product(a.begin(),a.end(), a.begin(),value_type(0));

        tensor_type var = (a+a)/2.0f; // std::complex<float>/int not allowed as expression is captured
        auto r2 = ublas::norm( var );

        BOOST_CHECK_EQUAL( c , r );
        BOOST_CHECK_EQUAL( std::sqrt( c ) , r2 );

    });
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_trans, value,  test_types, fixture )
{
    using namespace boost::numeric;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;

    for_each_tuple(static_extents,[&](auto const&, auto & n){
        using extents_type = typename std::decay<decltype(n)>::type;
        using tensor_type  = ublas::static_tensor<value_type, extents_type,layout_type>;
        
        constexpr auto const p = extents_type::_size;

        constexpr auto const s = ublas::static_product_v<extents_type>;
        auto aref = tensor_type();
        auto v    = value_type{};
        for(auto i = 0u; i < s; ++i, v+=1)
            aref[i] = v;

        
        using mp_list = boost::mp11::mp_pop_front< boost::mp11::mp_iota_c<p + 1> >;

        auto pi = mp_list_to_int_seq_t<mp_list> {};
        auto a = ublas::trans( aref, pi );
        
        for(auto i = 0ul; i < a.size(); i++){
            BOOST_CHECK( a[i] == aref[i]  );
        }

        constexpr auto const pfac = static_factorial_v<p> - 1;

        auto res_param = ublas::detail::static_for<0ul,pfac>([&](auto, auto perm){
            using perm_type = decltype(perm.second);
            using next_perm = static_next_permutation<perm_type>;
            using new_pi = mp_list_to_int_seq_t<next_perm>;
            return std::make_pair( ublas::trans( perm.first, new_pi{} ), next_perm{} );
        }, std::make_pair(a, mp_list{}) );

        auto res = ublas::detail::static_for<0ul,pfac>([&](auto, auto perm){
            using perm_type = decltype(perm.second);
            auto inverse_lm = [&](auto iter, auto pres){
                using iter_type = decltype(iter);
                using pres_type = decltype(pres);
                using pi_at = boost::mp11::mp_at<perm_type,iter_type>;
                constexpr auto const J = iter_type::value;
                using res_type = boost::mp11::mp_replace_at_c<pres_type, pi_at::value - 1, std::integral_constant<std::size_t, J + 1> >;
                return res_type{};
            };
            auto pi_inv = ublas::detail::type_static_for<0,p>(std::move(inverse_lm),perm_type{});
            using new_pi_inv = mp_list_to_int_seq_t<decltype(pi_inv)>;
            using prev_perm = static_prev_permutation<perm_type>;
            return std::make_pair( ublas::trans( perm.first, new_pi_inv{} ), prev_perm{});
        }, res_param);

        auto& ntensor = res.first;
        for(auto j = 0ul; j < a.size(); j++){
            BOOST_CHECK( ntensor[j] == aref[j]  );
        }
    });

}

BOOST_AUTO_TEST_SUITE_END()
