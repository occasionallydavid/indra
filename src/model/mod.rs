macro_rules! imported_model {
    ($name:ident) => {
        pub mod $name {
            #![allow(warnings, clippy::all)]
            include!(concat!(
                env!("OUT_DIR"),
                "/models/",
                stringify!($name),
                ".rs"
            ));
        }
    };
}

imported_model!(det);
imported_model!(rec);
imported_model!(order);
imported_model!(orient);
imported_model!(script);
imported_model!(rec_cjk);
imported_model!(mathcls);
imported_model!(math_enc);
imported_model!(math_dec_init);
imported_model!(math_dec_step);
