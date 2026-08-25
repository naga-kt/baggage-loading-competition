import numpy as np


# ============================================================
# 荷物の向き(orientation index)ごとの半径寸法
# シミュレーター側(utils.ORNS / get_half_ext)と完全に一致させる。
# 本エージェントは 0(そのまま) と 3(Z軸90度回転) の2種類のみ使用する
# = 「寝かせたまま(高さは変えない)」向きのみ。転倒/積み上げ不安定化を避けるため。
# ============================================================
def get_half_ext(length: float, width: float, height: float, orn_idx: int):
    if orn_idx == 0:
        return [length / 2, width / 2, height / 2]
    elif orn_idx == 1:
        return [length / 2, height / 2, width / 2]
    elif orn_idx == 2:
        return [height / 2, width / 2, length / 2]
    elif orn_idx == 3:
        return [width / 2, length / 2, height / 2]
    elif orn_idx == 4:
        return [width / 2, height / 2, length / 2]
    elif orn_idx == 5:
        return [height / 2, length / 2, width / 2]
    return [length / 2, width / 2, height / 2]


def quat_to_rotmat(q):
    """クオータニオン(x,y,z,w) -> 3x3回転行列"""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _aabb_overlap(a, b, eps: float = 0.002) -> bool:
    """2つのAABB (x0,x1,y0,y1,z0,z1) が重なっているか判定。
    epsだけ境界を広げて判定する(ギリギリ接するだけの"ほぼ重なり"も安全側で重なり扱いにする)。
    best_placementの探索が貪欲法であるがゆえに、しばしば障害物とのクリアランスが
    ほぼゼロの境界ぎりぎりの候補を選んでしまうため、その対策として境界判定に余裕を持たせる。
    """
    ax0, ax1, ay0, ay1, az0, az1 = a
    bx0, bx1, by0, by1, bz0, bz1 = b
    return (ax0 < bx1 + eps and ax1 > bx0 - eps and
            ay0 < by1 + eps and ay1 > by0 - eps and
            az0 < bz1 + eps and az1 > bz0 - eps)


def world_aabb_of_packed_item(item: dict):
    """
    settle後の実座標(pos/orn)から、その荷物のワールド座標系AABB(軸平行境界箱)を計算する。
    箱を回転させた8頂点から min/max を取るので、多少斜めに着地していても
    "はみ出さない側"に安全にくるむ(=保守的)。
    """
    pos = np.array(item['pos'], dtype=np.float64)
    orn = item['orn']
    hl, hw, hh = item['length'] / 2, item['width'] / 2, item['height'] / 2
    corners = np.array([[sx * hl, sy * hw, sz * hh]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    R = quat_to_rotmat(orn)
    world_corners = corners @ R.T + pos
    return world_corners.min(axis=0), world_corners.max(axis=0)


def extract_axis_bounds(n_vecs, points, tol: float = 1e-2):
    """
    container_list[i]['n_vecs'] / ['points'] (コンテナ内壁の各面の法線・代表点)から、
    軸平行な6面(x_min/x_max/y_min/y_max/z_min/z_max)と、
    LD3特有の斜めカット平面(軸に平行でない面)を自動判定して取り出す。

    ハードコードした長さ計算に頼らず実データから境界を求めることで、
    「ドア側は壁厚を引かない(開口部のため)」といった実装の細かな非対称性にも
    自動的に追従できる。
    """
    bounds = {}
    cut_plane = None  # (nx, nz, px, pz) 斜めカット平面(x-z平面内, y方向には一様)
    for n, p in zip(n_vecs, points):
        nx, ny, nz = n
        if abs(ny) > 1 - tol and abs(nx) < tol and abs(nz) < tol:
            bounds['y_max' if ny > 0 else 'y_min'] = p[1]
        elif abs(nz) > 1 - tol and abs(nx) < tol and abs(ny) < tol:
            bounds['z_max' if nz > 0 else 'z_min'] = p[2]
        elif abs(nx) > 1 - tol and abs(ny) < tol and abs(nz) < tol:
            bounds['x_max' if nx > 0 else 'x_min'] = p[0]
        else:
            # 軸に平行でない = LD3の斜めカット平面(1枚のみ想定)
            cut_plane = (nx, nz, p[0], p[2])
    return bounds, cut_plane


class ContainerState:
    """
    1台のコンテナについて、observationのcontainer_list[i]から
    毎回「その場で」再構築する高さグリッド(height-map)。
    前ステップの内部状態を持ち越さないので、物理シミュレーションの実際の結果と
    自分の内部状態がズレる(ドリフトする)心配がない。
    """

    def __init__(self, info: dict, list_pos: int, res: float = 0.04, safety: float = 0.01,
                 inclusion_margin: float = -0.01):
        self.list_pos = list_pos                      # env.step の container_idx に渡す実際の位置
        self.length = info['length']
        self.width = info['width']
        self.height = info['height']
        self.thickness = info['thickness']
        self.cut_x = info['cut_x']
        self.cut_y = info['cut_y']
        self.is_prioritized = info.get('is_prioritized', False)
        self.n_vecs = np.array(info['n_vecs'], dtype=np.float64)
        self.points = np.array(info['points'], dtype=np.float64)
        self.container_volume = info.get('volume', self.length * self.width * self.height)
        self.require_shelf = info.get('shelf', False)
        # local_to_global は x 座標だけ offset_x を加算する実装なので、
        # center[0] から offset_x を逆算できる
        self.offset_x = info['center'][0]
        self.buffer = info.get('buffer', 0.0)
        self._packed_items_raw = info.get('packed_items', [])
        self._transport_obstacles_cache = None

        self.res = res
        self.safety = safety
        # 厳密判定(check_inclusion_exact)で要求されるマージン。斜め平面のかさ上げ計算で使う。
        self.inclusion_margin = inclusion_margin
        # float32変換や再計算による丸め誤差で境界判定がフリップしないよう、
        # 内部の安全側計算にだけ使う微小な追加余裕(数値誤差対策。幾何的な意味はない)
        self._eps = 2e-3
        # ドア際キープアウトゾーンの幅。実機テストで繰り返し、ドアのすぐ手前に荷物が
        # 密集して後続の搬入経路そのものを塞ぐ致命的な失敗が確認されたため、
        # 優先/非優先を問わず、この距離以内には一切荷物を置かない。
        self.door_keepout = 0.25

        # 実データから境界を取得(手計算の式に頼らない -> ドア側の壁厚なし等の非対称性にも対応)
        bounds, cut_plane = extract_axis_bounds(self.n_vecs, self.points)
        self.x_min = bounds['x_min'] + safety + self._eps
        self.x_max = bounds['x_max'] - safety - self._eps
        self.y_min = bounds['y_min'] + safety + self._eps
        self.y_max = bounds['y_max'] - safety - self._eps
        self.z_floor = bounds['z_min'] + safety + self._eps
        self.z_ceiling = bounds['z_max'] - safety - self._eps
        self._x_min_raw = bounds['x_min']

        self.nx = max(1, int((self.x_max - self.x_min) / res))
        self.ny = max(1, int((self.y_max - self.y_min) / res))

        self.height_grid = np.full((self.nx, self.ny), self.z_floor, dtype=np.float64)
        self.soft_grid = np.zeros((self.nx, self.ny), dtype=bool)

        # 1) LD3の斜めカット角: 平面の式から列(x)ごとの実際の最低使用可能高さを計算し、
        #    矩形近似ではなく斜面なりに床を持ち上げる
        if cut_plane is not None:
            nx_c, nz_c, px_c, pz_c = cut_plane
            if abs(nz_c) > 1e-6:
                slope = -(nx_c / nz_c)  # min_z(margin=0) = pz_c + slope * (x - px_c)
                # check_inclusion_exact は nx*(x-px) + nz*(z-pz) <= inclusion_margin を要求する。
                # nzで割って z について解くと: z >= pz + (inclusion_margin - nx*(x-px)) / nz
                #                              = [margin=0の式] + inclusion_margin / nz
                # 平面が傾いている(|nz|<1)ほど、同じマージンを満たすのに必要な鉛直方向のかさ上げは
                # 大きくなる(margin/nz)。ここを一律safetyで代用すると傾きの分だけ不足するため、
                # 平面の傾きを反映したかさ上げ量を使う。さらに丸め誤差対策でepsも上乗せする。
                margin_z_offset = self.inclusion_margin / nz_c + self._eps / abs(nz_c)
                for ix in range(self.nx):
                    # 列内で最もmin_zが厳しくなる側のxを使う(離散化による過小評価を防ぐ)。
                    # slope<0(xが増えるほどmin_z減少)なら列の左端(x最小)が厳しい側、
                    # slope>0なら列の右端(x最大)が厳しい側になる。
                    x_left = self.x_min + ix * res
                    x_right = x_left + res
                    x_worst = x_left if slope <= 0 else x_right
                    min_z = pz_c + slope * (x_worst - px_c) + margin_z_offset
                    self.height_grid[ix, :] = np.maximum(self.height_grid[ix, :], min_z)

        # 2) 常設の「小さい棚」(small_shelf): cut_x帯・コンテナ高さの中央付近(厚みthickness)を
        #    横切る薄い梁のような固定障害物で、require_shelfの有無に関わらず必ず存在する。
        #    形状には反映されないため、この帯だけは中央高さまでしか使わないよう天井を制限する。
        shelf_x0 = self._x_min_raw
        shelf_x1 = self._x_min_raw + self.cut_x
        shelf_z_low = self.height / 2 - safety
        ix0 = max(0, int((shelf_x0 - self.x_min) / res))
        ix1 = min(self.nx, int(np.ceil((shelf_x1 - self.x_min) / res)))
        self.shelf_band_ceiling = np.full(self.nx, self.z_ceiling, dtype=np.float64)
        if ix1 > ix0:
            self.shelf_band_ceiling[ix0:ix1] = np.minimum(self.shelf_band_ceiling[ix0:ix1], shelf_z_low)

        if self.require_shelf:
            # 大きい方の棚(require_shelf=True時のみ)は未対応。安全側に倒すため、
            # コンテナ全体の天井を棚の高さ付近までに制限しておく(過度な過信を避ける)。
            # 詳細な形状はcontainers.pyのcreate()参照。将来的な改善ポイント。
            self.shelf_band_ceiling[:] = np.minimum(self.shelf_band_ceiling, self.height / 2 - safety)

        self.filled_volume = 0.0
        for item in info.get('packed_items', []):
            self._register_item(item)

    # -------------------- 内部ユーティリティ --------------------

    def _x_to_idx(self, x):
        return int(round((x - self.x_min) / self.res))

    def _y_to_idx(self, y):
        return int(round((y - self.y_min) / self.res))

    def _idx_to_x(self, ix):
        return self.x_min + ix * self.res

    def _idx_to_y(self, iy):
        return self.y_min + iy * self.res

    def _register_item(self, item: dict):
        """既に積み付けられている荷物1個を高さグリッドに反映する"""
        if item.get('pos') is None or item.get('orn') is None:
            return
        (wx0, wy0, wz0), (wx1, wy1, wz1) = world_aabb_of_packed_item(item)
        lx0, lx1 = wx0 - self.offset_x, wx1 - self.offset_x  # ローカルx (yとzはoffsetなし)
        ix0, ix1 = self._x_to_idx(lx0), self._x_to_idx(lx1) + 1
        iy0, iy1 = self._y_to_idx(wy0), self._y_to_idx(wy1) + 1
        ix0, ix1 = max(0, ix0), min(self.nx, ix1)
        iy0, iy1 = max(0, iy0), min(self.ny, iy1)
        if ix1 > ix0 and iy1 > iy0:
            self.height_grid[ix0:ix1, iy0:iy1] = np.maximum(
                self.height_grid[ix0:ix1, iy0:iy1], wz1
            )
            self.soft_grid[ix0:ix1, iy0:iy1] = bool(item.get('is_soft', False))
        self.filled_volume += item['length'] * item['width'] * item['height']

    def filled_ratio(self) -> float:
        if self.container_volume <= 0:
            return 1.0
        return min(1.0, self.filled_volume / self.container_volume)

    # -------------------- 配置探索 --------------------

    def best_placement(self, item: dict, prefer_front: bool, top_k: int = 5):
        """
        この荷物をこのコンテナに置ける候補位置を、スコアの良い順に最大top_k件探索する。
        戻り値: dict(score, local_center, orn_idx, cells, top_z)のリスト(スコア昇順)

        1件だけでなく複数候補を返すのは、近似グリッド上ではベストに見えても、
        (LD3の斜めカット面など)厳密なinclusion判定ではわずかにはみ出して弾かれる
        ケースがあり得るため。その場合でも次点候補を試せるようにする防御的設計。
        """
        found = []
        candidates = ((0, item['length'], item['width']),
                      (3, item['width'], item['length']))

        for orn_idx, fl, fw in candidates:
            fcx = max(1, int(np.ceil(fl / self.res)))
            fcy = max(1, int(np.ceil(fw / self.res)))
            if fcx > self.nx or fcy > self.ny:
                continue

            for ix in range(0, self.nx - fcx + 1):
                # この footprint が跨る列のうち、最も厳しい天井制限(棚帯)を採用
                local_ceiling = float(self.shelf_band_ceiling[ix:ix + fcx].min())

                for iy in range(0, self.ny - fcy + 1):
                    # ドア際キープアウトゾーン: 荷物の手前側の辺がドアから一定距離以内に
                    # 入る配置は、優先/非優先を問わず一切許可しない。
                    # 全ての荷物はここ(y=y_min)を起点に搬入されるため、ここを塞ぐと
                    # 後続の荷物が物理的に入れなくなる(実機で繰り返し確認された致命的な失敗パターン)。
                    near_edge_y = self._idx_to_y(iy)
                    if near_edge_y < self.y_min + self.door_keepout:
                        continue

                    region = self.height_grid[ix:ix + fcx, iy:iy + fcy]
                    base_z = float(region.max())
                    top_z = base_z + item['height']
                    if top_z > local_ceiling:
                        continue

                    # 支持率: footprintのうち、実際にbase_z付近(=荷物の底面)で
                    # 支えられているセルの割合。base_zより明確に低い(=宙に浮いている)
                    # セルが多い配置は、物理演算でグラつく/転倒するリスクが高いため避ける。
                    support_tol = max(self.res, 0.02)  # 2cm(またはグリッド解像度)未満の段差は「支持あり」とみなす
                    supported = region >= (base_z - support_tol)
                    support_ratio = float(supported.mean())

                    # 支持率が低すぎる(半分未満しか支えられていない)候補はそもそも危険なので除外
                    min_support_ratio = 0.6
                    if support_ratio < min_support_ratio:
                        continue

                    # 硬い荷物をソフト手荷物の真上に直接載せることを避ける(可能な限り)
                    soft_penalty = 0.0
                    if not item.get('is_soft', False):
                        region_soft = self.soft_grid[ix:ix + fcx, iy:iy + fcy]
                        if region_soft[region >= base_z - 1e-6].any():
                            soft_penalty = 500.0

                    y_center_idx = iy + fcy / 2.0
                    x_center = self._idx_to_x(ix) + fl / 2.0

                    if prefer_front:
                        # 優先手荷物: 手前(y小)を優先しつつ、中央の搬入レーン(x=0付近)を
                        # 塞がないよう側面寄り(|x|が大きい側)に誘導する。
                        # x_center=0で最大、コンテナ端に近づくほど小さくなるペナルティ。
                        side_bias = (self.length / 2.0 - abs(x_center)) * 30.0
                        y_pref = y_center_idx + side_bias
                    else:
                        # 非優先: 奥(y大 = ドアから遠い)を優先 -> 後続荷物の搬入経路を塞ぎにくい
                        y_pref = -y_center_idx

                    # ドア際(y最小ライン)に、幅広くx方向に張り出す配置を避ける。
                    # 通行帯を塞ぐ「壁」になりやすいのは、ドアのすぐ際(iy=0付近)かつ
                    # x方向に幅がある(fcxが大きい)配置なので、両者の積でペナルティを課す。
                    door_adjacency = max(0.0, 1.0 - iy * self.res / 0.15)  # ドアから15cm以内で最大1.0
                    door_block_penalty = door_adjacency * fcx * self.res * 400.0

                    # 支持率が低いほどペナルティを課す(1.0=満点で支持されている場合ペナルティ0)
                    support_penalty = (1.0 - support_ratio) * 2000.0

                    # 高さ(安定性・衝突リスク)を最優先、次に支持率(転倒防止)、
                    # 次にドア際ブロッキング回避、コンテナ負荷バランス、
                    # 最後にy方向/側面の好み、という優先順位
                    score = (round(top_z / self.res) * 10000.0
                             + support_penalty
                             + door_block_penalty
                             + self.filled_ratio() * 50.0
                             + y_pref
                             + soft_penalty)

                    cx = self._idx_to_x(ix) + fl / 2.0
                    cy = self._idx_to_y(iy) + fw / 2.0
                    cz = base_z + item['height'] / 2.0
                    found.append({
                        'score': score,
                        'local_center': (cx, cy, cz),
                        'orn_idx': orn_idx,
                        'cells': (ix, ix + fcx, iy, iy + fcy),
                        'top_z': top_z,
                        'support_ratio': support_ratio,
                    })

        if not found:
            return []
        found.sort(key=lambda d: d['score'])
        return found[:top_k]

    def check_inclusion_exact(self, local_center, half_lwh, margin: float = -0.01) -> bool:
        """
        シミュレーター側 validator.check_inclusion と同じ式で、
        コンテナの厳密な内壁形状(n_vecs/points, LD3の斜めカットも含む)に
        対して収まっているかを最終確認する。
        """
        gx = local_center[0] + self.offset_x
        gy = local_center[1]
        gz = local_center[2]
        target = np.array([gx, gy, gz], dtype=np.float64)
        dots = (self.n_vecs * (target - self.points)).sum(axis=1) \
            + np.abs(self.n_vecs) @ np.array(half_lwh, dtype=np.float64)
        return bool(np.all(dots <= margin))

    def check_transport_path_approx(self, item: dict, local_center, orn_idx: int,
                                     start_margin: float = 0.01, start_z: float = 0.08,
                                     safety_margin: float = 0.03, ceiling_margin: float = 0.018) -> bool:
        """
        validator.check_transport_path (pybulletで実際に少しずつ動かして干渉判定する処理) の近似版。
        物理エンジンを使わず、y方向移動 -> x方向移動の2セグメントを軸平行AABBの掃引箱として扱い、
        既配置の荷物および常設の"小さい棚"(small_shelf)との重なりを判定する。

        注意: これはあくまで幾何的な近似(荷物本体の回転や物理的な押し出しは考慮しない)。
        start_margin/start_z/safety_margin/ceiling_marginはvalidator設定値だが、
        policyはvalidatorの設定を直接受け取れないため、sample_config.jsonの値を既定値として使う。
        """
        half_lwh = get_half_ext(item['length'], item['width'], item['height'], orn_idx)
        hx, hy, hz = half_lwh
        L, W, H, T = self.length, self.width, self.height, self.thickness
        target_x, target_y, target_z = local_center

        # 1. スタートx位置のクランプ(斜めカット領域を避けて、その右側から搬入する)
        x_min = -L / 2 + T + self.cut_x + hx + start_margin
        x_max = L / 2 - T - hx - start_margin
        rel_x = min(max(target_x, x_min), x_max)

        # 2. 直置き判定(床/棚面に十分近ければ浮かせずそのまま横移動)
        resting_surfaces = [T, H / 2 + T + self.buffer]
        ceiling_surfaces = [H / 2 + self.buffer, H + self.buffer - T]
        effective_start_z = start_z
        bottom_z = target_z - hz
        for r_z in resting_surfaces:
            if 0 <= (bottom_z - r_z) <= 0.05:
                effective_start_z = 0.0
                break
        top_z = target_z + hz
        if effective_start_z > 0.0:
            for c_z in ceiling_surfaces:
                clearance = c_z - top_z
                if 0 <= clearance < (effective_start_z + ceiling_margin):
                    effective_start_z = max(0.0, clearance - ceiling_margin - 0.0005)
                    break

        rel_z = min(H + self.buffer - T - hz - start_margin, target_z + effective_start_z)

        obstacles = self._transport_obstacles(safety_margin)

        # セグメント1: (rel_x, y, rel_z) を y=-W/2 -> target_y へ
        y_lo, y_hi = sorted((-W / 2, target_y))
        seg1 = (rel_x - hx, rel_x + hx, y_lo - hy, y_hi + hy, rel_z - hz, rel_z + hz)
        for obs in obstacles:
            if _aabb_overlap(seg1, obs):
                return False

        # セグメント2: (x, target_y, rel_z) を rel_x -> target_x へ
        x_lo, x_hi = sorted((rel_x, target_x))
        seg2 = (x_lo - hx, x_hi + hx, target_y - hy, target_y + hy, rel_z - hz, rel_z + hz)
        for obs in obstacles:
            if _aabb_overlap(seg2, obs):
                return False

        return True

    def _transport_obstacles(self, safety_margin: float):
        """既配置の荷物 + 常設の小さい棚(small_shelf)を、ローカル座標系のAABBリストとして返す(safety_margin分だけ膨張済み)"""
        if self._transport_obstacles_cache is not None:
            return self._transport_obstacles_cache

        obstacles = []
        for item in self._packed_items_raw:
            if item.get('pos') is None or item.get('orn') is None:
                continue
            (wx0, wy0, wz0), (wx1, wy1, wz1) = world_aabb_of_packed_item(item)
            lx0, lx1 = wx0 - self.offset_x, wx1 - self.offset_x
            obstacles.append((lx0 - safety_margin, lx1 + safety_margin,
                               wy0 - safety_margin, wy1 + safety_margin,
                               wz0 - safety_margin, wz1 + safety_margin))

        # 常設の"小さい棚"(cut_x帯の中央高さ付近を横切る固定の梁)。require_shelfの有無に関わらず必ず存在する。
        L, W, H, T = self.length, self.width, self.height, self.thickness
        shelf_cx = -L / 2 + self.cut_x / 2 + T
        shelf_hx, shelf_hy, shelf_hz = self.cut_x / 2, W / 2 - T, T / 2
        shelf_cz = H / 2 + T / 2 + self.buffer
        obstacles.append((shelf_cx - shelf_hx - safety_margin, shelf_cx + shelf_hx + safety_margin,
                           -shelf_hy - safety_margin, shelf_hy + safety_margin,
                           shelf_cz - shelf_hz - safety_margin, shelf_cz + shelf_hz + safety_margin))

        self._transport_obstacles_cache = obstacles
        return obstacles


class Agent:
    """
    高さグリッド(height-map)ベースの逐次配置エージェント。

    設計方針:
      1. 観測(container_list)から毎回グリッドを作り直すため、内部状態のドリフトがない
      2. コンテナ内壁の形状(n_vecs/points)を実データから解析し、LD3特有の斜めカット角を
         矩形近似ではなく斜面として、常設の「小さい棚」障害物も天井制限として考慮する
      3. 荷物の搬入は「ドア側からy方向へ直進 -> x方向へスライド」という物理を踏まえ、
         通常は奥(ドアから遠い側)から埋めるように誘導し、将来の搬入経路を塞がない
      4. 向きは0(そのまま)/3(Z軸90度回転)のみに限定し、寝かせた姿勢を維持(転倒対策)
      5. 最終的に厳密なinclusion判定(n_vecs/points)を行い、近似誤差による無効配置を防ぐ
      6. 見えているプール内の全荷物 × 全コンテナ × 全向きを総当たりし、
         最もスコアの良い組み合わせを選ぶ(貪欲法)

    既知の制約(V2以降での改善ポイント):
      - require_shelf=True の大きい棚は簡易的に「その帯は高さ半分までしか使わない」
        という保守的な近似で処理しており、精密な形状は反映していない
      - 搬入経路(check_transport_path)そのものはシミュレートしていないため、
        y方向の奥優先配置というヒューリスティックで衝突リスクを下げているに過ぎない
    """

    def __init__(self, module_path: str):
        self.res = 0.04       # グリッド解像度[m]
        self.safety = 0.01    # 内壁からの追加安全マージン[m]
        self.inclusion_margin = -0.01

    def get_init_states(self, init_states: dict):
        # 実際の状態構築は毎ステップ policy() 内でobservationから行うため、
        # ここではコンテナ数などの軽い確認のみ行う
        self.num_containers = len(init_states.get('container_list', []))
        return True

    def optimize(self, item_list: list):
        """
        オフライン最適化: 事前に全荷物が分かっている場合の搬入順序を決める。
        - 優先手荷物を先頭に寄せる(早めに確保しておきたいため)
        - 同グループ内では体積の大きい順(大物を先に置いた方が空間の断片化を防げる)
        """
        def sort_key(it):
            vol = it.get('volume', it['length'] * it['width'] * it['height'])
            return (0 if it.get('is_prioritized') else 1, -vol)

        sorted_items = sorted(item_list, key=sort_key)
        return [it['index'] for it in sorted_items]

    def _try_candidates(self, container_state, item: dict, candidates: list):
        """候補リストを順に厳密判定(inclusion + 搬入経路)にかけ、最初に通ったものを返す。
        全滅した場合はNone。"""
        for placement in candidates:
            half_lwh = get_half_ext(item['length'], item['width'], item['height'],
                                     placement['orn_idx'])
            if not container_state.check_inclusion_exact(placement['local_center'], half_lwh,
                                                           margin=self.inclusion_margin):
                continue
            if not container_state.check_transport_path_approx(item, placement['local_center'],
                                                                 placement['orn_idx']):
                continue
            return placement
        return None

    def policy(self, observation: dict):
        pool_list = observation.get('pool_list', [])
        container_infos = observation.get('container_list', [])

        if not pool_list or not container_infos:
            return self._fallback_action()

        containers = [ContainerState(info, list_pos=i, res=self.res, safety=self.safety,
                                      inclusion_margin=self.inclusion_margin)
                      for i, info in enumerate(container_infos)]

        best = None
        for pool_idx, item in enumerate(pool_list):
            prefer_front = bool(item.get('is_prioritized', False))
            for c in containers:
                # 1段階目: 少数の候補で高速に探す(通常はここで十分見つかる)
                candidates = c.best_placement(item, prefer_front=prefer_front, top_k=40)
                found_here = self._try_candidates(c, item, candidates)
                if found_here is None:
                    # 2段階目: 高さ層の浅いところに搬入経路OKな候補が無かった場合のみ、
                    # コスト覚悟で全件探索にエスカレーションする(policy_timeoutに注意しつつ)
                    all_candidates = c.best_placement(item, prefer_front=prefer_front,
                                                        top_k=1_000_000)
                    found_here = self._try_candidates(c, item, all_candidates)
                if found_here is not None:
                    placement = found_here
                    if best is None or placement['score'] < best['score']:
                        best = {
                            'score': placement['score'],
                            'pool_idx': pool_idx,
                            'container': c,
                            'local_center': placement['local_center'],
                            'orn_idx': placement['orn_idx'],
                        }

        if best is None:
            return self._fallback_action(pool_list)

        action = {
            'item_idx': best['pool_idx'],
            'container_idx': best['container'].list_pos,
            'place_pos': np.array(best['local_center'], dtype=np.float32),
            'orientation': best['orn_idx'],
        }
        return action

    def _fallback_action(self, pool_list=None):
        """置き場所が全く見つからない(=コンテナが実質満杯)場合の最終手段"""
        return {
            'item_idx': 0,
            'container_idx': 0,
            'place_pos': np.array([0.0, 0.0, 0.5], dtype=np.float32),
            'orientation': 0,
        }
