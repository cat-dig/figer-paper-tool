# 一键打包脚本
# 用法: 在 PowerShell 中运行 .\build.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "论文组图工具 - 一键打包" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 激活conda环境
Write-Host "[1/4] 激活 jiechu_build 环境..." -ForegroundColor Yellow
& "D:\Anaconda\shell\condabin\conda-hook.ps1"
conda activate jiechu_build

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 环境激活失败！" -ForegroundColor Red
    exit 1
}

# 清理旧文件
Write-Host "[2/4] 清理旧的打包文件..." -ForegroundColor Yellow
if (Test-Path ".\dist") {
    Remove-Item -Path ".\dist" -Recurse -Force
    Write-Host "✓ 已清理 dist 目录" -ForegroundColor Green
}
if (Test-Path ".\build") {
    Remove-Item -Path ".\build" -Recurse -Force
    Write-Host "✓ 已清理 build 目录" -ForegroundColor Green
}

# 运行PyInstaller
Write-Host "[3/4] 开始打包..." -ForegroundColor Yellow
pyinstaller --clean PaperFigure.spec

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 打包失败！" -ForegroundColor Red
    exit 1
}

# 检查结果
Write-Host "[4/4] 检查打包结果..." -ForegroundColor Yellow
if (Test-Path ".\dist\PaperFigure\PaperFigure.exe") {
    $size = (Get-ChildItem ".\dist\PaperFigure" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    $size = [math]::Round($size, 2)
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ 打包成功！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "程序位置: .\dist\PaperFigure\PaperFigure.exe" -ForegroundColor Cyan
    Write-Host "文件夹大小: $size MB" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "提示: 发布时需要整个 PaperFigure 文件夹，不要只复制 exe 文件" -ForegroundColor Yellow
    Write-Host ""
    
    # 询问是否立即运行
    $response = Read-Host "是否立即运行测试？(Y/N)"
    if ($response -eq 'Y' -or $response -eq 'y') {
        Write-Host "启动程序..." -ForegroundColor Cyan
        Start-Process ".\dist\PaperFigure\PaperFigure.exe"
    }
} else {
    Write-Host "❌ 未找到打包后的exe文件！" -ForegroundColor Red
    exit 1
}
