use icicle_core::curve::Affine;
use icicle_core::traits::FieldImpl;
use memmap::{Mmap, MmapOptions};
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom};
use std::mem;
use std::path::Path;

use crate::zkey::ZKey;
use crate::{F, G1, G2};

const GROTH16_PROTOCOL_ID: u32 = 1;

#[derive(Clone, Debug)]
pub struct Wtsn {
    pub n8: usize,
    pub q: F,
    pub n_witness: usize,
}

#[derive(Clone, Debug)]
pub struct Section {
    pub p: u64,
    pub size: u64,
}

#[derive(Debug)]
pub struct FileWrapper {
    pub file: File,
    pub reading_section: Option<Section>,
    pub mmap: Mmap,
}

impl FileWrapper {
    pub fn new(file: File) -> io::Result<Self> {
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self {
            file,
            reading_section: None,
            mmap,
        })
    }

    pub fn read_bin_file(
        file_name: &str,
        expected_type: &str,
        max_version: u32,
    ) -> io::Result<(File, Vec<Vec<Section>>)> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(file_name)
            .unwrap();

        let mut buf = [0; 4];
        file.read_exact(&mut buf).unwrap();
        let read_type = String::from_utf8(buf.to_vec()).expect("Invalid UTF-8 sequence");

        if read_type != expected_type {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: Invalid File format", file_name),
            ));
        }

        let mut version_buf = [0; 4];
        file.read_exact(&mut version_buf).unwrap();
        let version = u32::from_le_bytes(version_buf);

        if version > max_version {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Version not supported",
            ));
        }

        let mut sections_count_buf = [0; 4];
        file.read_exact(&mut sections_count_buf).unwrap();
        let n_sections = u32::from_le_bytes(sections_count_buf);

        let mut sections: Vec<Vec<Section>> = vec![Vec::new(); (n_sections + 1) as usize];

        for _ in 0..n_sections {
            let mut ht_buf = [0; 4];
            file.read_exact(&mut ht_buf).unwrap();
            let ht = u32::from_le_bytes(ht_buf) as usize;

            let mut hl_buf = [0; 8];
            file.read_exact(&mut hl_buf).unwrap();
            let hl = u64::from_le_bytes(hl_buf);

            let current_pos = file.stream_position().unwrap();
            sections[ht].push(Section {
                p: current_pos,
                size: hl,
            });

            file.seek(SeekFrom::Current(hl as i64)).unwrap();
        }

        Ok((file, sections))
    }

    pub fn save_json_file<P: AsRef<Path>, T: Serialize>(
        path: P,
        data: &T,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, data)?;
        Ok(())
    }

    pub fn start_read_unique_section(
        &mut self,
        sections: &[Vec<Section>],
        id_section: usize,
    ) -> io::Result<()> {
        if self.reading_section.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Already reading a section",
            ));
        }

        if sections.get(id_section).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Missing section {}", id_section),
            ));
        }

        if sections[id_section].len() > 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Section Duplicated {}", id_section),
            ));
        }

        self.file
            .seek(SeekFrom::Start(sections[id_section][0].p))
            .unwrap();
        self.reading_section = Some(sections[id_section][0].clone());

        Ok(())
    }

    pub fn end_read_section(&mut self, no_check: bool) -> io::Result<()> {
        let section = self
            .reading_section
            .take()
            .ok_or(io::Error::new(io::ErrorKind::InvalidInput, "Not reading"))?;
        if !no_check && self.file.stream_position()? - section.p != section.size {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Size mismatch"));
        }
        Ok(())
    }

    pub fn read_big_int(&mut self, n8: usize, pos: Option<u64>) -> io::Result<F> {
        let mut buff = vec![0u8; n8];
        if let Some(pos) = pos {
            self.file.seek(SeekFrom::Start(pos)).unwrap();
        }
        self.file.read_exact(&mut buff).unwrap();
        Ok(F::from_bytes_le(&buff))
    }

    pub fn read_wtns_header(&mut self, sections: &[Vec<Section>]) -> io::Result<Wtsn> {
        self.start_read_unique_section(sections, 1).unwrap();
        let n8 = self.read_u32_le().unwrap() as usize;
        let q = self.read_big_int(n8, None).unwrap();
        let n_witness = self.read_u32_le().unwrap() as usize;
        self.end_read_section(false).unwrap();

        Ok(Wtsn { n8, q, n_witness })
    }

    pub fn read_u32_le(&mut self) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        self.file.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    pub fn read_section(
        &self,
        sections: &[Vec<Section>],
        id_section: usize,
    ) -> Result<&[u8], io::Error> {
        let start = sections[id_section][0].p as usize;
        let end = start + sections[id_section][0].size as usize;

        Ok(&self.mmap[start..end])
    }

    pub fn read_zkey_header(&mut self, sections: &[Vec<Section>]) -> io::Result<ZKey> {
        self.start_read_unique_section(sections, 1).unwrap();
        let protocol_id = self.read_u32_le().unwrap();
        self.end_read_section(false).unwrap();

        match protocol_id {
            GROTH16_PROTOCOL_ID => ZKey::read_header_groth16(self, sections),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Protocol not supported",
            )),
        }
    }

    pub fn read_g1(&mut self) -> G1 {
        let mut x = [0u8; 32];
        self.file.read_exact(&mut x).unwrap();

        let x: [u32; 8] = unsafe { mem::transmute(x) };

        let mut y = [0u8; 32];
        self.file.read_exact(&mut y).unwrap();

        let y: [u32; 8] = unsafe { mem::transmute(y) };

        Affine::from_limbs(x, y)
    }

    pub fn read_g2(&mut self) -> G2 {
        let mut x = [0u8; 64];
        self.file.read_exact(&mut x).unwrap();

        let x: [u32; 16] = unsafe { std::mem::transmute(x) };

        let mut y = [0u8; 64];
        self.file.read_exact(&mut y).unwrap();

        let y: [u32; 16] = unsafe { std::mem::transmute(y) };

        Affine::from_limbs(x, y)
    }
}
