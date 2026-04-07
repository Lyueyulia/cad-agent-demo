def execute_cad_code(code: str):
    """
    执行 Agent 生成的 CadQuery 代码并保存为 STEP 文件。
    """
    try:
        # 延迟导入，避免未安装 cadquery 时模块导入即失败
        import cadquery as cq  # pyright: ignore[reportMissingImports]

        # 定义一个局部作用域，确保代码能访问到 cadquery
        loc = {}

        # 执行生成的代码字符串
        exec(code, {"cq": cq}, loc)
        
        # 约定 Agent 必须将最终结果赋值给变量 'result'
        if "result" in loc:
            obj = loc["result"]
            # 导出为工业标准的 STEP 格式
            cq.exporters.export(obj, 'part.step')
            # 同时导出 STL（用于网页 3D 预览）
            try:
                cq.exporters.export(obj, "part.stl")
            except Exception:
                # 不影响 STEP 的交付
                pass
            return "Success: 文件已生成为 part.step"
        else:
            return "Error: 代码运行成功，但未定义 'result' 变量。"
            
    except Exception as e:
        return f"Error during execution: {str(e)}"