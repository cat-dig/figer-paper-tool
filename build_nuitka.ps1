# 一键打包脚本 - Nuitka版本
# 用法: 在 PowerShell 中运行 .\build_nuitka.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "论文组图工具 - Nuitka打包" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 激活conda环境
Write-Host "[1/5] 激活 jiechu_build 环境..." -ForegroundColor Yellow
& "D:\Anaconda\shell\condabin\conda-hook.ps1"
conda activate jiechu_build

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 环境激活失败！" -ForegroundColor Red
    exit 1
}

# 检查Nuitka是否安装
Write-Host "[2/5] 检查Nuitka..." -ForegroundColor Yellow
$nuitkaCheck = python -m nuitka --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Nuitka未安装，正在安装..." -ForegroundColor Yellow
    pip install nuitka
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Nuitka安装失败！" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✓ Nuitka已就绪" -ForegroundColor Green

# 清理旧文件
Write-Host "[3/5] 清理旧的打包文件..." -ForegroundColor Yellow
if (Test-Path ".\dist") {
    Remove-Item -Path ".\dist" -Recurse -Force
    Write-Host "✓ 已清理 dist 目录" -ForegroundColor Green
}
if (Test-Path ".\build") {
    Remove-Item -Path ".\build" -Recurse -Force
    Write-Host "✓ 已清理 build 目录" -ForegroundColor Green
}
if (Test-Path ".\论文组图工具.dist") {
    Remove-Item -Path ".\论文组图工具.dist" -Recurse -Force
    Write-Host "✓ 已清理 论文组图工具.dist 目录" -ForegroundColor Green
}
if (Test-Path ".\论文组图工具.build") {
    Remove-Item -Path ".\论文组图工具.build" -Recurse -Force
    Write-Host "✓ 已清理 论文组图工具.build 目录" -ForegroundColor Green
}

# 运行Nuitka打包
Write-Host "[4/5] 开始Nuitka打包（首次运行会自动下载C编译器，可能需要30分钟+）..." -ForegroundColor Yellow
Write-Host "提示: Nuitka会自动下载MinGW64编译器（约500MB），请耐心等待..." -ForegroundColor Cyan

# 检查是否有图标文件并构建参数
$nuitkaArgs = @(
    "--standalone",
    "--onefile",
    "--mingw64",
    "--assume-yes-for-downloads",
    "--enable-plugin=pyside6",
    "--include-data-dir=config=config",
    "--include-data-dir=layouts=layouts",
    "--windows-console-mode=disable",
    "--output-dir=dist",
    "--output-filename=论文组图工具.exe",
    "--company-name=Research Tools",
    "--product-name=论文组图工具",
    "--file-version=2.0.0",
    "--product-version=2.0.0",
    "--file-description=论文组图工具 - 科研图片布局与标注",
    "--copyright=Copyright (c) 2026",
    "--remove-output",
    "--show-progress",
    "--show-memory",
    "Figure_paper.py"
)

if (Test-Path ".\icon.ico") {
    $nuitkaArgs = @("--windows-icon-from-ico=icon.ico") + $nuitkaArgs
    Write-Host "✓ 找到图标文件，将使用自定义图标" -ForegroundColor Green
} else {
    Write-Host "⚠ 未找到icon.ico，将使用默认图标" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "开始编译..." -ForegroundColor Green
python -m nuitka @nuitkaArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 打包失败！请查看错误信息" -ForegroundColor Red
    exit 1
}

# 检查结果
Write-Host "[5/5] 检查打包结果..." -ForegroundColor Yellow
if (Test-Path ".\dist\论文组图工具.exe") {
    $size = (Get-ChildItem ".\dist\论文组图工具.exe").Length / 1MB
    $size = [math]::Round($size, 2)
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ 打包成功！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "程序位置: .\dist\论文组图工具.exe" -ForegroundColor Cyan
    Write-Host "文件大小: $size MB" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "提示: Nuitka打包为单文件可执行程序，可以直接分发" -ForegroundColor Yellow
    Write-Host ""
    
    # 询问是否立即运行
    $response = Read-Host "是否立即运行测试？(Y/N)"
    if ($response -eq 'Y' -or $response -eq 'y') {
        Write-Host "启动程序..." -ForegroundColor Cyan
        Start-Process ".\dist\论文组图工具.exe"
    }
} else {
    Write-Host "❌ 未找到打包后的exe文件！" -ForegroundColor Red
    exit 1
}
