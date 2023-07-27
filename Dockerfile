FROM rust:1.69

COPY . .

RUN cargo build --release

CMD [ "./target/release/tideapi" ]
EXPOSE 8080
