# 슬라이서용 C++ 패치

`support_bead_close_packing.cpp` 는 그 자체로 컴파일되는 파일이 아니라,
PrusaSlicer / OrcaSlicer 계열의 `src/libslic3r/Support/SupportMaterial.cpp`
익명 네임스페이스 안에 들어갈 헬퍼 모음입니다.

파이썬 도구와 **완전히 같은 배치 규칙**을 씁니다. 파이썬 쪽에서 파라미터를
확정한 뒤 그 값을 그대로 여기에 옮기고, `--dump-json` 으로 뽑은 좌표와
슬라이서 결과를 대조하는 순서를 권합니다.

## 원본에서 바뀐 곳

원본 패치 대비 셋만 다릅니다.

1. **격자 원점 고정.** 원본은 `get_extents(safe_polygons)`, 즉 그 층 폴리곤의
   bbox 를 격자 기준으로 썼습니다. 층마다 단면이 달라지면 원점이 흔들려서,
   시프트를 정확히 줘도 아래층 hollow 에 안 떨어집니다. 행 인덱스도 층 로컬
   카운터가 아니라 전역 격자 인덱스로 바꿔야 행 패리티가 유지됩니다.
2. **`% 2` → `% stagger_period`.** 원본의 시프트 `(pitch_x/2, pitch_y/3)` 는
   이미 hollow 좌표라서, 층 번호만큼 누적하면 A→B→C 순환이 됩니다.
   period 2 는 HCP, 3 은 FCC 입니다.
3. **`mm3_per_mm` 을 구 부피에서 역산.** 원본은 `flow.mm3_per_mm()` 에
   `flow_multiplier` 를 곱했는데, 그러면 구 크기가 세그먼트 길이에
   끌려다녀서 구형을 유지할 수 없습니다.

나머지(이름, 구조, snake order, 얇은 인터페이스 섬 shrink 폴백)는 원본 그대로입니다.

## 호출부에서 해야 할 일

파일 맨 아래 주석에 정리해 뒀습니다. 요약하면,

```cpp
// 1. 격자 원점을 오브젝트당 한 번만 설정
contact_params.grid_origin = support_bead_grid_origin(object_bbox);
body_params.grid_origin    = contact_params.grid_origin;

// 2. 첫 서포터 층은 원래대로 sheath / no_sort 경로 유지 (베드 접착)
if (! first_support_layer && ! sheath && ! no_sort) { ... }

// 3. 프로파일 층높이가 충전 기하와 맞는지 확인
const double h_req = body_params.layer_height_mm();
if (std::abs(layer_height - h_req) > 0.02)
    BOOST_LOG_TRIVIAL(warning) << "Bead support: layer height "
        << layer_height << " mm should be " << h_req << " mm";
```

3번을 빼먹으면 조용히 충전이 깨집니다. 층높이가 기하와 안 맞으면 구가 아래층에
안 닿거나 뭉개지는데, 슬라이싱은 정상적으로 끝나기 때문에 출력해 보기 전까지
알 수 없습니다.
