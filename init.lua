vim.pack.add({
    -- 'https://github.com/morhetz/gruvbox',
    'https://github.com/tiagovla/tokyodark.nvim',
    'https://github.com/honza/vim-snippets',
    'https://github.com/neovim/nvim-lspconfig',
    'https://github.com/hrsh7th/cmp-nvim-lsp',
    'https://github.com/hrsh7th/cmp-buffer',
    'https://github.com/hrsh7th/cmp-path',
    'https://github.com/hrsh7th/cmp-cmdline',
    'https://github.com/hrsh7th/nvim-cmp',
    'https://github.com/m4xshen/autoclose.nvim',
    'https://github.com/nvim-neo-tree/neo-tree.nvim',
    'https://github.com/nvim-lua/plenary.nvim',
    'https://github.com/MunifTanjim/nui.nvim',
    'https://github.com/nvim-tree/nvim-web-devicons',
    'https://github.com/lervag/vimtex',
})

vim.o.tabstop = 4
vim.o.number = true
vim.o.relativenumber = true
vim.o.shiftwidth = 4
vim.o.expandtab = true
vim.o.hlsearch = false

vim.g.vimtex_view_method = 'sioyek'

-- ctags optimization
-- set tags=tags;

vim.keymap.set('n', '<C-h>', '<C-w>h')
vim.keymap.set('n', '<C-j>', '<C-w>j')
vim.keymap.set('n', '<C-k>', '<C-w>k')
vim.keymap.set('n', '<C-l>', '<C-w>l')
vim.keymap.set('i', 'jj', '<Esc>')

vim.keymap.set('n', '<F5>', ':Neotree toggle reveal<CR><C-w>p')

vim.cmd.colorscheme('tokyodark')

require('neo-tree').setup({
    close_if_last_window = true,
    enable_git_status = true,
    enable_refresh_on_write = false,

    window = {
        position = "left", -- left, right, top, bottom, float, current
        width = 30,
        height = 15,
    },
})

vim.api.nvim_create_autocmd('VimEnter', {
  callback = function()
    vim.cmd('Neotree toggle reveal position=left')
    vim.defer_fn(function()
      vim.cmd("wincmd p")
    end, 0)
  end,
})
--[[
imap jj <Esc>
map <C-h> <C-w>h
map <C-j> <C-w>j
map <C-k> <C-w>k
map <C-l> <C-w>l
nmap <F5> :NERDTreeToggle<CR><C-w>p

let g:NERDTreeWinSize=18
let g:indentLine_char = '┊'
let g:indentLine_color_term = 8
colorscheme gruvbox
set background=dark

autocmd BufEnter * lcd %:p:h
autocmd VimEnter * NERDTree
autocmd VimEnter * wincmd p
]]

-- does not run in the selected venv
-- autocmd FileType python map <buffer> <F9> :w<CR>:exec '!python3 shellescape(@%, 1)<CR>
-- autocmd FileType python imap <buffer> <F9> <esc>:w<CR>:exec '!python3' shellescape(@%, 1)<CR>

--[[
augroup python_run
  autocmd!
  autocmd FileType python nnoremap <buffer> <F9> :w<CR>:!python3 %<CR>
  autocmd FileType python inoremap <buffer> <F9> <Esc>:w<CR>:!python3 %<CR>
augroup END

autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree")
      \ && b:NERDTree.isTabTree()) | q | endif
]]


-- autoclose
require('autoclose').setup()


local cmp = require('cmp')

local capabilities = require('cmp_nvim_lsp').default_capabilities()

-- PYTHON LSP
vim.lsp.config('pylsp', {
  capabilities = capabilities,
  settings = {
      pylsp = {
          plugins = {
              ruff = {enabled = true,}
          }
      }
  }
})
vim.lsp.enable('pylsp')

-- CPP
vim.lsp.config('clangd', {
    capabilities = capabilities,
})
vim.lsp.enable('clangd')

vim.diagnostic.config({
  virtual_text = true,
  signs = true,
  underline = true,
  update_in_insert = false,
  severity_sort = true,
})

-- AUTOCOMPLETE
cmp.setup({
  snippet = {
    expand = function(args)
      vim.fn["vsnip#anonymous"](args.body)
    end,
  },

  mapping = cmp.mapping.preset.insert({
    ['<Tab>'] = cmp.mapping.select_next_item(),
    ['<S-Tab>'] = cmp.mapping.select_prev_item(),

    ['<CR>'] = cmp.mapping.confirm({ select = true }),

    ['<C-Space>'] = cmp.mapping.complete(),
  }),

  sources = {
    { name = 'nvim_lsp' },
    { name = 'buffer' },
    { name = 'vsnip' },
  }
})

-- LSP KEYMAPS
vim.keymap.set('n', 'gd', vim.lsp.buf.definition)
vim.keymap.set('n', 'gD', vim.lsp.buf.declaration)
vim.keymap.set('n', 'K', vim.lsp.buf.hover)

vim.keymap.set('n', '[d', vim.diagnostic.goto_prev)
vim.keymap.set('n', ']d', vim.diagnostic.goto_next)
vim.keymap.set('n', '<leader>e', vim.diagnostic.open_float)
