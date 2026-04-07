import os
from pathlib import Path
import sys
import re
import base64
import json

import streamlit as st  # pyright: ignore[reportMissingImports]
import streamlit.components.v1 as components  # pyright: ignore[reportMissingImports]

from agent_graph import app


APP_TITLE = "工业级 AI CAD 多轮对话助手"
STEP_PATH = Path("part.step")
STL_PATH = Path("part.stl")
GLTF_PATH = Path("part.gltf")


def _init_session():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("agent_history", [])
    st.session_state.setdefault("last_turn_id", 0)
    st.session_state.setdefault("last_parser_push_turn_id", -1)
    st.session_state.setdefault("last_system_push_turn_id", -1)
    st.session_state.setdefault("last_inputs", None)
    st.session_state.setdefault("last_result", None)


def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 工业级 AI 建模助手 (Connecting Rod Demo)")
st.caption("由 LangGraph + Gemini + CadQuery 驱动的多轮对话式生成式设计引擎")

with st.expander("运行环境信息（调试）", expanded=False):
    st.code(sys.executable, language="text")
    st.caption("建议使用 cad311 启动：")
    st.code("& \"C:\\Users\\liang\\anaconda3\\envs\\cad311\\python.exe\" -m streamlit run app.py", language="powershell")

_init_session()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("💬 对话输入")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("用自然语言描述你要建模的零件与约束（支持追问/微调）")

with col_right:
    st.subheader("🔎 解析结果 / 参数协商")
    st.caption("Parser 会识别零件类型、提取参数，并提出缺失参数的追问问题。")

if user_text:
    if app is None:
        with col_left:
            st.error("工作流未初始化成功，请先安装并检查 langgraph 依赖。")
        st.stop()

    st.session_state.last_turn_id += 1
    turn_id = st.session_state.last_turn_id

    _append_message("user", user_text)

    with st.spinner("多 Agent 正在工作中（Parser → Planner → Coder → Executor）..."):
        # 避免误把旧的 part.step 当作本轮输出
        if STEP_PATH.exists():
            try:
                STEP_PATH.unlink()
            except Exception:
                pass
        if STL_PATH.exists():
            try:
                STL_PATH.unlink()
            except Exception:
                pass
        if GLTF_PATH.exists():
            try:
                GLTF_PATH.unlink()
            except Exception:
                pass
        bin_path = Path("part.bin")
        if bin_path.exists():
            try:
                bin_path.unlink()
            except Exception:
                pass
        inputs = {
            "input": user_text,
            "chat_history": st.session_state.messages,
            "history": st.session_state.agent_history,
            "iterations": 0,
            "errors": "",
        }
        st.session_state.last_inputs = inputs
        result = app.invoke(inputs)
        st.session_state.last_result = result

    st.session_state.agent_history = result.get("history", []) or []
    metadata = result.get("metadata") or {}
    status = result.get("status") or ""
    errors = result.get("errors") or ""

    questions = metadata.get("clarifying_questions") if isinstance(metadata, dict) else None
    # PRD：Parser 追问消息推送至聊天框，且避免重复推送
    if status == "need_clarification" and questions:
        if st.session_state.last_parser_push_turn_id != turn_id:
            q_text = "\n".join([f"- {q}" for q in questions])
            _append_message("assistant", f"【Parser 追问】我需要你补充/确认以下信息后才能继续建模：\n{q_text}")
            st.session_state.last_parser_push_turn_id = turn_id
    else:
        # PRD：系统最终通知（成功/失败）推送到聊天框，且避免重复推送
        if st.session_state.last_system_push_turn_id != turn_id:
            if errors:
                _append_message("assistant", "【系统通知】本轮执行失败，我已保留错误信息（见右侧）。你可以补充约束或点击右侧“一键重试”。")
            else:
                _append_message("assistant", "【系统通知】本轮任务已完成。如需微调请继续描述修改点（例如“厚度加大到 20mm”）。")
            st.session_state.last_system_push_turn_id = turn_id

    # 关键：本轮新增了聊天消息，需要立即 rerun 才能显示在左侧聊天线程
    st.rerun()

    # rerun 后由下面的统一渲染逻辑展示右侧面板/结果

with col_left:
    if not st.session_state.messages:
        st.info("请在聊天框输入：例如“做一个连杆，中心距150mm，大头50mm，小头20mm”。或“做一个带 6 孔的法兰盘，PCD 80mm”。")

# --- 统一渲染（避免 rerun 导致右侧不显示） ---
last_result = st.session_state.get("last_result") or {}
metadata = last_result.get("metadata") or {}
status = last_result.get("status") or ""
errors = last_result.get("errors") or ""
questions = metadata.get("clarifying_questions") if isinstance(metadata, dict) else None

with col_right:
    st.subheader("🔎 动态控制面板 (Generative Dashboard)")
    st.caption("参数实时看板 + 执行链路状态 + 结果/错误反馈")

    if isinstance(metadata, dict):
        part_type = metadata.get("part_type", "unknown")
        conf = float(metadata.get("confidence") or 0.0)
        missing = metadata.get("missing_params") or []
        c1, c2, c3 = st.columns(3)
        c1.metric("part_type", str(part_type))
        c2.metric("confidence", f"{conf:.2f}")
        c3.metric("missing_params", str(len(missing)))

    st.subheader("🧩 Parser 输出（metadata）")
    st.json(metadata)

    with st.status("Agent 链路状态", expanded=True) as s:
        hist_join = "".join(st.session_state.agent_history)
        s.write("Parser: OK" if "Parser 已解析" in hist_join else "Parser: …")
        if status == "need_clarification":
            s.write("Planner: Blocked (need clarification)")
            s.write("Coder: Blocked (need clarification)")
            s.write("Executor: Blocked (need clarification)")
            s.update(state="running")
        else:
            s.write("Planner: " + ("OK" if "Planner 已生成建模计划" in hist_join else "…"))
            s.write("Coder: " + ("OK" if "Code Agent 已生成建模脚本" in hist_join else "…"))
            s.write("Executor: " + ("OK" if not errors else "Error"))
            s.update(state="error" if errors else "complete")

    if status == "need_clarification" and questions:
        st.warning("需要你确认/补充信息后才能继续建模：")
        for q in questions:
            st.write("- " + str(q))
        if errors and ("RESOURCE_EXHAUSTED" in str(errors) or "429" in str(errors)):
            retry_s = None
            m = re.search(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", str(errors))
            if m:
                try:
                    retry_s = float(m.group(1))
                except Exception:
                    retry_s = None
            st.info(
                "检测到 Gemini 429 限流/配额耗尽，本轮已自动降级为规则解析以保证可用。"
                + (f"建议等待约 {retry_s:.0f}s 后再试，或降低调用频率/更换模型。" if retry_s else "建议稍后再试，或降低调用频率/更换模型。")
            )
    elif errors:
        st.error("执行报错（见下方 errors），你可以补充约束或点击“一键重试”。")
    elif last_result:
        st.success("任务完成。")

    with st.expander("🧭 Agent 历史轨迹（history）", expanded=True):
        st.write(last_result.get("history", []))
    with st.expander("🧯 错误信息（errors）", expanded=False):
        st.code(errors or "", language="text")

    if errors and st.session_state.last_inputs:
        if st.button("🔁 一键重试", use_container_width=True):
            with st.spinner("正在重试..."):
                retry_inputs = dict(st.session_state.last_inputs)
                retry_inputs["history"] = st.session_state.agent_history
                retry_inputs["chat_history"] = st.session_state.messages
                retry_inputs["iterations"] = 0
                retry_inputs["errors"] = ""
                st.session_state.last_inputs = retry_inputs
                st.session_state.last_result = app.invoke(retry_inputs)
            st.rerun()

    st.subheader("📦 STEP 输出")
    if status != "need_clarification" and (not errors) and STEP_PATH.exists():
        st.info("STEP 文件已生成：`part.step`")
        with STEP_PATH.open("rb") as f:
            st.download_button(
                label="📥 下载 STEP 模型",
                data=f,
                file_name="part.step",
                mime="application/octet-stream",
            )
    else:
        st.caption("当前没有新的 STEP 输出（可能还在补问阶段，或建模失败，或是旧文件已被隐藏）。")

    st.subheader("🧱 3D 实时预览（STL）")
    if status != "need_clarification" and (not errors) and STL_PATH.exists():
        try:
            import pyvista as pv  # pyright: ignore[reportMissingImports]

            mesh = pv.read(str(STL_PATH))
            # 避免 trame/server/multiprocessing 在 Windows + Streamlit 下的句柄/序列化问题：
            # 使用 VTK 的 glTF 导出（plotter.export_gltf），再用 three.js 在前端渲染。
            pl = pv.Plotter(off_screen=True, window_size=(600, 420))
            pl.add_mesh(mesh, color="#d0d4db", smooth_shading=True)
            pl.view_isometric()
            pl.set_background("#ffffff")
            pl.export_gltf(str(GLTF_PATH))

            gltf_text = GLTF_PATH.read_text(encoding="utf-8")
            gltf_json = json.loads(gltf_text)

            # pyvista 默认 inline_data=True，会把 buffer 写成 data:...;base64,...（Windows 下 path 不能用这个做文件名）
            # 若不是内嵌（inline_data=False），则再走 blob URL 兜底。
            buffers = gltf_json.get("buffers") or []
            uris: list[str] = []
            if isinstance(buffers, list):
                for b in buffers:
                    if isinstance(b, dict) and isinstance(b.get("uri"), str):
                        uris.append(b["uri"])
            all_inline = (len(uris) > 0) and all(u.startswith("data:") for u in uris)

            gltf_b64 = base64.b64encode(gltf_text.encode("utf-8")).decode("ascii")

            # 若不是全内嵌，尝试把第一个外部 buffer 读出来塞进 blob（尽量兼容）
            bin_b64 = ""
            if not all_inline:
                uri0 = uris[0] if uris else "part.bin"
                bin_path = GLTF_PATH.with_name(uri0)
                if bin_path.exists():
                    bin_b64 = base64.b64encode(bin_path.read_bytes()).decode("ascii")

            html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      html, body {{ margin: 0; padding: 0; background: #fff; }}
      #msg {{ font: 12px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial; color:#333; padding:8px 10px; }}
      #c {{ width: 100%; height: 440px; display: block; }}
    </style>
  </head>
  <body>
    <div id="msg">3D 预览加载中…（如果长时间空白，请看这里的错误提示）</div>
    <canvas id="c"></canvas>
    <script type="module">
      const msg = document.getElementById("msg");
      function setMsg(t) {{ msg.textContent = t; }}

      async function importFrom(base) {{
        // base 示例：
        // - https://cdn.jsdelivr.net/npm/three@0.164.1
        // - https://unpkg.com/three@0.164.1
        const THREE = await import(`${{base}}/build/three.module.js`);
        const oc = await import(`${{base}}/examples/jsm/controls/OrbitControls.js`);
        const gl = await import(`${{base}}/examples/jsm/loaders/GLTFLoader.js`);
        return {{ THREE, OrbitControls: oc.OrbitControls, GLTFLoader: gl.GLTFLoader }};
      }}

      async function boot() {{
        const bases = [
          "https://cdn.jsdelivr.net/npm/three@0.164.1",
          "https://unpkg.com/three@0.164.1",
          "https://esm.sh/three@0.164.1",
        ];

        let THREE, OrbitControls, GLTFLoader;
        for (const base of bases) {{
          try {{
            const mod = await importFrom(base);
            THREE = mod.THREE;
            OrbitControls = mod.OrbitControls;
            GLTFLoader = mod.GLTFLoader;
            break;
          }} catch (e) {{
            console.warn("three import failed from", base, e);
          }}
        }}
        if (!THREE || !OrbitControls || !GLTFLoader) {{
          setMsg("无法加载 three.js（CDN 可能被拦截）。请切换网络/VPN，或改为本地打包 three.js 静态资源。");
          return;
        }}

        const canvas = document.getElementById("c");
        let renderer;
        try {{
          renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: true }});
        }} catch (e) {{
          console.error(e);
          setMsg("WebGL 初始化失败（可能被浏览器/系统禁用）。");
          return;
        }}
        renderer.setPixelRatio(window.devicePixelRatio || 1);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);

        const camera = new THREE.PerspectiveCamera(45, 2, 0.01, 1000);
        camera.position.set(2.5, 2.0, 2.5);

        const controls = new OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;

        const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
        hemi.position.set(0, 1, 0);
        scene.add(hemi);
        const dir = new THREE.DirectionalLight(0xffffff, 0.8);
        dir.position.set(5, 8, 4);
        scene.add(dir);

        function b64ToUint8(b64) {{
          const bin = atob(b64);
          const len = bin.length;
          const arr = new Uint8Array(len);
          for (let i = 0; i < len; i++) arr[i] = bin.charCodeAt(i);
          return arr;
        }}

        const gltfJson = JSON.parse(atob("{gltf_b64}"));
        const binB64 = "{bin_b64}";
        if (binB64 && gltfJson.buffers && gltfJson.buffers[0] && !(String(gltfJson.buffers[0].uri || "").startsWith("data:"))) {{
          const binBytes = b64ToUint8(binB64);
          const blob = new Blob([binBytes], {{ type: "application/octet-stream" }});
          const blobUrl = URL.createObjectURL(blob);
          gltfJson.buffers[0].uri = blobUrl;
        }}

        const loader = new GLTFLoader();
        const gltfStr = JSON.stringify(gltfJson);
        setMsg("3D 解析中…");
        loader.parse(
          gltfStr,
          "",
          (gltf) => {{
            const root = gltf.scene || gltf.scenes?.[0];
            scene.add(root);

            // 自动居中和缩放到视野内
            const box = new THREE.Box3().setFromObject(root);
            const size = new THREE.Vector3();
            box.getSize(size);
            const center = new THREE.Vector3();
            box.getCenter(center);
            root.position.sub(center);

            const maxDim = Math.max(size.x, size.y, size.z) || 1;
            camera.position.set(maxDim * 1.2, maxDim * 0.9, maxDim * 1.2);
            camera.lookAt(0, 0, 0);
            controls.update();
            setMsg("3D 预览就绪（可拖拽旋转/滚轮缩放）");
          }},
          (err) => {{
            console.error(err);
            setMsg("3D 解析失败（请打开浏览器控制台查看具体错误）。");
          }}
        );

        function resizeRendererToDisplaySize() {{
          const width = canvas.clientWidth;
          const height = canvas.clientHeight;
          const needResize = canvas.width !== width || canvas.height !== height;
          if (needResize) {{
            renderer.setSize(width, height, false);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
          }}
          return needResize;
        }}

        function render() {{
          resizeRendererToDisplaySize();
          controls.update();
          renderer.render(scene, camera);
          requestAnimationFrame(render);
        }}
        render();
      }}
      boot();
    </script>
  </body>
</html>
"""
            components.html(html, height=460, scrolling=False)
        except Exception as exc:
            st.caption(f"3D 预览加载失败：{exc}（你仍可下载 STEP）。")
    else:
        st.caption("暂无 STL 预览（仅在建模成功后生成 `part.stl`）。")