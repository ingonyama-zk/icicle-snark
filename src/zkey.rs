use crate::{
    ProtocolId,
    groth16::zkey::{ZKeyHeader as Groth16ZKeyHeader},
    file_wrapper::{FileWrapper, Section},
    G1, G2,
};
use icicle_core::curve::Affine;
use std::io::{Error, ErrorKind, Result, Read};
use std::mem;

// NOTE: Specific to bn254
pub const W: [&str; 30] = [
    "0x0000000000000000000000000000000000000000000000000000000000000001",
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
    "0x30644e72e131a029048b6e193fd841045cea24f6fd736bec231204708f703636",
    "0x2b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e80",
    "0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b",
    "0x09c532c6306b93d29678200d47c0b2a99c18d51b838eeb1d3eed4c533bb512d0",
    "0x1418144d5b080fcac24cdb7649bdadf246a6cb2426e324bedb94fb05118f023a",
    "0x16e73dfdad310991df5ce19ce85943e01dcb5564b6f24c799d0e470cba9d1811",
    "0x07b0c561a6148404f086204a9f36ffb0617942546750f230c893619174a57a76",
    "0x0f1ded1ef6e72f5bffc02c0edd9b0675e8302a41fc782d75893a7fa1470157ce",
    "0x06fd19c17017a420ebbebc2bb08771e339ba79c0a8d2d7ab11f995e1bc2e5912",
    "0x027a358499c5042bb4027fd7a5355d71b8c12c177494f0cad00a58f9769a2ee2",
    "0x0931d596de2fd10f01ddd073fd5a90a976f169c76f039bb91c4775720042d43a",
    "0x006fab49b869ae62001deac878b2667bd31bf3e28e3a2d764aa49b8d9bbdd310",
    "0x2d965651cdd9e4811f4e51b80ddca8a8b4a93ee17420aae6adaa01c2617c6e85",
    "0x2d1ba66f5941dc91017171fa69ec2bd0022a2a2d4115a009a93458fd4e26ecfb",
    "0x00eeb2cb5981ed45649abebde081dcff16c8601de4347e7dd1628ba2daac43b7",
    "0x1bf82deba7d74902c3708cc6e70e61f30512eca95655210e276e5858ce8f58e5",
    "0x19ddbcaf3a8d46c15c0176fbb5b95e4dc57088ff13f4d1bd84c6bfa57dcdc0e0",
    "0x2260e724844bca5251829353968e4915305258418357473a5c1d597f613f6cbd",
    "0x26125da10a0ed06327508aba06d1e303ac616632dbed349f53422da953337857",
    "0x1ded8980ae2bdd1a4222150e8598fc8c58f50577ca5a5ce3b2c87885fcd0b523",
    "0x1ad92f46b1f8d9a7cda0ceb68be08215ec1a1f05359eebbba76dde56a219447e",
    "0x0210fe635ab4c74d6b7bcf70bc23a1395680c64022dd991fb54d4506ab80c59d",
    "0x0c9fabc7845d50d2852e2a0371c6441f145e0db82e8326961c25f1e3e32b045b",
    "0x2a734ebb326341efa19b0361d9130cd47b26b7488dc6d26eeccd4f3eb878331a",
    "0x1067569af1ff73b20113eff9b8d89d4a605b52b63d68f9ae1c79bd572f4e9212",
    "0x049ae702b363ebe85f256a9f6dc6e364b4823532f6437da2034afc4580928c44",
    "0x2a3c09f0a58a7e8500e0a7eb8ef62abc402d111e41112ed49bd61b6e725b19f0",
    "0x2260e724844bca5251829353968e4915305258418357473a5c1d597f613f6cbd",
];

pub enum ZKeyHeader {
    Groth16(Groth16ZKeyHeader),
    // Plonk(PlonkZKeyHeader),
}

pub struct ZKey {
    pub file: FileWrapper,
    pub protocol_id: ProtocolId,
    pub sections: Vec<Vec<Section>>,
    pub header: ZKeyHeader,
}

impl ZKey {
    pub fn load(zkey_path: &str) -> Result<Self> {
        let (mut file, sections) = FileWrapper::read_bin_file(zkey_path, "zkey", 2).unwrap();
        file.start_read_unique_section(&sections, 1).unwrap();
        let protocol_id = file.read_u32_le().unwrap();
        if let Some(protocol_id) = ProtocolId::from_u32(protocol_id) {
            file.end_read_section(false).unwrap();
            let header = Self::read_header(&mut file, protocol_id, &sections);
            Ok(Self {
                file,
                protocol_id,
                sections,
                header,
            })
        } else {
            //  TODO: change to use std::error::Error and std::error::Result
            Err(Error::new(
                ErrorKind::InvalidData,
                "Protocol not supported",
            ))
        }
    }

    fn read_header(fw: &mut FileWrapper, protocol_id: ProtocolId, sections: &[Vec<Section>]) -> ZKeyHeader {
        match protocol_id {
            ProtocolId::Groth16 => ZKeyHeader::Groth16(Groth16ZKeyHeader::read_header(fw, sections)),
        }
    }
}


pub fn read_g1(fw: &mut FileWrapper) -> G1 {
    let mut x = [0u8; 32];
    fw.file.read_exact(&mut x).unwrap();

    let x: [u32; 8] = unsafe { mem::transmute(x) };

    let mut y = [0u8; 32];
    fw.file.read_exact(&mut y).unwrap();

    let y: [u32; 8] = unsafe { mem::transmute(y) };

    Affine::from_limbs(x, y)
}

pub fn read_g2(fw: &mut FileWrapper) -> G2 {
    let mut x = [0u8; 64];
    fw.file.read_exact(&mut x).unwrap();

    let x: [u32; 16] = unsafe { std::mem::transmute(x) };

    let mut y = [0u8; 64];
    fw.file.read_exact(&mut y).unwrap();

    let y: [u32; 16] = unsafe { std::mem::transmute(y) };

    Affine::from_limbs(x, y)
}
