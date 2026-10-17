use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// A recognized token. CTC output ids are kept as ids until the line is joined;
/// CJK output is already a character and is kept as text.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Id { id: usize, x: f32 },
    Char { text: String, x: f32 },
}

impl Token {
    pub fn x(&self) -> f32 {
        match self {
            Self::Id { x, .. } | Self::Char { x, .. } => *x,
        }
    }
}

#[derive(Debug, Deserialize)]
struct VocabFile {
    #[serde(default)]
    blank: usize,
    chars: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Vocab {
    chars: Vec<String>,
    index: HashMap<String, usize>,
    has_commands: bool,
}

impl Vocab {
    pub fn from_json(text: &str) -> Result<Self> {
        let file: VocabFile = serde_json::from_str(text).context("invalid vocabulary JSON")?;
        if file.blank != 0 {
            bail!("only blank=0 vocabularies are supported");
        }
        let has_commands = file.chars.iter().any(|token| token.chars().count() > 1);
        let index = file
            .chars
            .iter()
            .enumerate()
            .map(|(index, token)| (token.clone(), index + 1))
            .collect();
        Ok(Self {
            chars: file.chars,
            index,
            has_commands,
        })
    }

    pub fn chars(&self) -> &[String] {
        &self.chars
    }

    pub fn contains_char(&self, ch: char) -> bool {
        self.index.contains_key(&ch.to_string())
    }

    pub fn space_id(&self) -> Option<usize> {
        self.index.get(" ").copied()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let tokens: Vec<&str> = ids
            .iter()
            .filter_map(|&id| id.checked_sub(1).and_then(|index| self.chars.get(index)))
            .map(String::as_str)
            .collect();
        if !self.has_commands {
            return tokens.concat();
        }

        let mut out = String::new();
        for (index, token) in tokens.iter().enumerate() {
            out.push_str(token);
            let next = tokens.get(index + 1).copied().unwrap_or("");
            if token.len() > 2
                && token.starts_with('\\')
                && token[1..].chars().all(char::is_alphabetic)
                && next.chars().next().is_some_and(char::is_alphabetic)
            {
                out.push(' ');
            }
        }
        out
    }

    pub fn decode_tokens(&self, tokens: &[Token]) -> String {
        if tokens
            .iter()
            .any(|token| matches!(token, Token::Char { .. }))
        {
            let joined: String = tokens
                .iter()
                .map(|token| match token {
                    Token::Char { text, .. } => text.as_str(),
                    Token::Id { id, .. } => id
                        .checked_sub(1)
                        .and_then(|index| self.chars.get(index))
                        .map(String::as_str)
                        .unwrap_or(""),
                })
                .collect();
            return joined.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        let ids: Vec<usize> = tokens
            .iter()
            .filter_map(|token| match token {
                Token::Id { id, .. } => Some(*id),
                Token::Char { .. } => None,
            })
            .collect();
        self.decode(&ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        Vocab::from_json(r#"{"blank":0,"chars":[" ","A","B","\\alpha","x"]}"#).unwrap()
    }

    #[test]
    fn decodes_ids_and_repairs_latex_spacing() {
        let vocab = vocab();
        assert_eq!(vocab.decode(&[2, 3]), "AB");
        assert_eq!(vocab.decode(&[4, 5]), "\\alpha x");
    }

    #[test]
    fn decodes_cjk_character_tokens() {
        let vocab = vocab();
        let tokens = vec![
            Token::Char {
                text: "你".into(),
                x: 1.0,
            },
            Token::Char {
                text: "好".into(),
                x: 2.0,
            },
        ];
        assert_eq!(vocab.decode_tokens(&tokens), "你好");
    }
}
