# Segurança e publicação

## Segredos

O projeto não requer segredo para acessar o checkpoint e dataset públicos usados por padrão. Nunca versione:

- `.env` ou variantes locais;
- tokens do GitHub ou Hugging Face;
- chaves privadas, certificados cliente ou arquivos de credenciais;
- URLs contendo usuário, senha ou token;
- dumps de configuração do sistema ou caches de autenticação.

O `.gitignore` cobre padrões comuns, mas não substitui revisão. Antes de cada publicação:

```bash
git status --short
git diff --cached --name-only
git diff --cached
```

Use também um scanner dedicado, como Gitleaks, em ambientes onde esteja disponível. Se um segredo entrar em um commit, apenas removê-lo no commit seguinte não basta: revogue/rotacione o segredo e reescreva o histórico quando apropriado.

## Dados

Os artefatos versionados derivam do SST-2 e não incluem textos brutos no estado atual; os CSVs contêm índices e medidas numéricas. Ao adaptar para dados privados:

- não versione exemplos brutos nem embeddings sem avaliação de privacidade;
- restrinja acesso aos artefatos e ao cache;
- defina retenção e anonimização;
- trate embeddings como dados potencialmente reveladores;
- valide licença e consentimento.

## Dependências e modelos

Dependências e modelos são código/dados de terceiros. Fixe revisões em produção, verifique procedência e acompanhe vulnerabilidades. `trust_remote_code` não é habilitado pelo projeto e não deve ser ativado sem revisão do repositório remoto.

## Repositório privado

Privacidade do repositório reduz exposição, mas não torna seguro versionar credenciais. Confirme visibilidade privada após a criação, mantenha proteção de conta/2FA e conceda acesso mínimo a colaboradores e automações.
